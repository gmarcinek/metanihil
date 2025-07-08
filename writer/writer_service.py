import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import re
from dotenv import load_dotenv
load_dotenv()

from .models import ChunkData, ChunkStatus, SearchResult, ProcessingStats, TOCEntry
from .persistence import ChunkPersistence
from .faiss_manager import FAISSManager
from .embedder import ChunkEmbedder


class WriterService:
    """Main service orchestrating chunks, embeddings, and search - unified API"""
    
    def __init__(self, storage_dir: str = "output/writer_storage", 
                 embedding_model: str = "text-embedding-3-small"):
        self.storage_dir = Path(storage_dir)
        self.embedding_model = embedding_model
        
        # Initialize components
        self.persistence = ChunkPersistence(storage_dir=str(self.storage_dir))
        self.faiss_manager = FAISSManager(storage_dir=str(self.storage_dir))
        self.embedder = ChunkEmbedder(model=embedding_model)
        
        # Load chunks into memory
        self.chunks = self.persistence.load_all_chunks()
        
        # Sync embeddings
        self._sync_embeddings_on_startup()
        
        print(f"📚 WriterService initialized: {len(self.chunks)} chunks loaded")
    
    def _sync_embeddings_on_startup(self):
        """Sync embeddings between persistence and FAISS on startup"""
        chunks_with_faiss = self.faiss_manager.get_all_chunk_ids_with_embeddings()
        chunks_in_storage = set(self.chunks.keys())
        
        # Clean up orphaned FAISS embeddings
        orphaned_embeddings = chunks_with_faiss - chunks_in_storage
        if orphaned_embeddings:
            self.faiss_manager.cleanup_orphaned_embeddings(chunks_in_storage)
        
        print(f"🔄 Embedding sync: {len(chunks_with_faiss)} in FAISS, {len(chunks_in_storage)} in storage")
    
    # ==================== TOC OPERATIONS ====================
    
    def parse_toc_content(self, toc_content: str) -> List[ChunkData]:
        """Parse TOC content and create chunk data"""
        toc_pattern = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')
        chunks = []
        
        for line in toc_content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            match = toc_pattern.match(line)
            if not match:
                continue
            
            hierarchical_id = match.group(1)
            title = match.group(2).strip()
            
            # Create TOC entry and convert to chunk
            toc_entry = TOCEntry(
                hierarchical_id=hierarchical_id,
                title=title,
                parent_hierarchical_id=self._extract_parent_id(hierarchical_id),
                level=len(hierarchical_id.split('.'))
            )
            
            chunk_id = self._generate_chunk_id()
            chunk = toc_entry.to_chunk_data(chunk_id)
            chunks.append(chunk)
        
        return chunks
    
    def save_toc_chunks(self, chunks: List[ChunkData]) -> int:
        """Save TOC chunks to storage"""
        saved_count = self.persistence.save_chunks(chunks)
        
        # Update in-memory cache
        for chunk in chunks:
            self.chunks[chunk.id] = chunk
        
        return saved_count
    
    def embed_toc_chunks(self, chunk_ids: Optional[List[str]] = None, force: bool = False) -> int:
        """Generate embeddings for TOC chunks (titles only initially)"""
        if chunk_ids:
            chunks_to_embed = [self.chunks[cid] for cid in chunk_ids if cid in self.chunks]
        else:
            chunks_to_embed = [chunk for chunk in self.chunks.values() 
                             if chunk.status == ChunkStatus.NOT_STARTED]
        
        if not chunks_to_embed:
            return 0
        
        print(f"📝 Embedding {len(chunks_to_embed)} TOC chunks")
        
        # Generate embeddings (use_content=False for TOC-only)
        embedding_results = self.embedder.embed_chunks_batch(chunks_to_embed, use_content=False)
        
        embedded_count = 0
        for chunk, embedding in embedding_results:
            if embedding is not None:
                # Update chunk
                chunk.embedding = embedding
                chunk.update_timestamp()
                
                # Add to FAISS
                self.faiss_manager.add_chunk_embedding(chunk, embedding)
                
                # Save to storage
                self.persistence.save_chunk(chunk)
                
                embedded_count += 1
        
        # Save FAISS index
        self.faiss_manager.save_index()
        
        print(f"✅ Embedded {embedded_count} TOC chunks")
        return embedded_count
    
    # ==================== CHUNK OPERATIONS ====================
    
    def get_chunks_by_status(self, status: ChunkStatus) -> List[ChunkData]:
        """Get chunks by status"""
        return [chunk for chunk in self.chunks.values() if chunk.status == status]
    
    def get_chunks_batch(self, status: ChunkStatus, limit: int) -> List[ChunkData]:
        """Get limited batch of chunks by status"""
        matching_chunks = self.get_chunks_by_status(status)
        matching_chunks.sort(key=lambda x: x.hierarchical_id)
        return matching_chunks[:limit]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkData]:
        """Get chunk by ID"""
        return self.chunks.get(chunk_id)
    
    def get_chunk_by_hierarchical_id(self, hierarchical_id: str) -> Optional[ChunkData]:
        """Get chunk by hierarchical ID"""
        for chunk in self.chunks.values():
            if chunk.hierarchical_id == hierarchical_id:
                return chunk
        return None
    
    def update_chunk_content(self, chunk_id: str, content: str, summary: str = None) -> bool:
        """Update chunk content and regenerate embedding"""
        chunk = self.get_chunk_by_id(chunk_id)
        if not chunk:
            return False
        
        # Update content
        chunk.content = content
        if summary:
            chunk.summary = summary
        chunk.status = ChunkStatus.COMPLETED
        chunk.update_timestamp()
        
        # Regenerate embedding with content
        embedding = self.embedder.embed_chunk(chunk, use_content=True)
        if embedding is not None:
            chunk.embedding = embedding
            self.faiss_manager.add_chunk_embedding(chunk, embedding)
        
        # Save to storage
        self.persistence.save_chunk(chunk)
        self.faiss_manager.save_index()
        
        return True
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete chunk from all systems"""
        if chunk_id not in self.chunks:
            return False
        
        # Remove from FAISS
        self.faiss_manager.remove_chunk_embedding(chunk_id)
        
        # Remove from persistence
        self.persistence.delete_chunk(chunk_id)
        
        # Remove from memory
        del self.chunks[chunk_id]
        
        return True
    
    # ==================== SEARCH OPERATIONS ====================
    
    def search_chunks_semantic(self, query: str, max_results: int = 10, 
                             min_similarity: float = 0.7) -> List[SearchResult]:
        """Semantic search using embeddings"""
        # Generate query embedding
        query_embedding = self.embedder.embed_text_query(query)
        if query_embedding is None:
            return []
        
        # Search FAISS
        faiss_results = self.faiss_manager.search_similar_chunks(
            query_embedding, max_results, min_similarity
        )
        
        # Convert to SearchResult objects
        search_results = []
        for chunk_id, similarity in faiss_results:
            chunk = self.get_chunk_by_id(chunk_id)
            if chunk:
                search_results.append(SearchResult(
                    chunk=chunk,
                    similarity_score=similarity,
                    match_type="semantic"
                ))
        
        return search_results
    
    def search_chunks_text(self, query: str, max_results: int = 10) -> List[ChunkData]:
        """Text-based search in titles and content"""
        query_lower = query.lower()
        results = []
        
        for chunk in self.chunks.values():
            # Check title
            if query_lower in chunk.title.lower():
                results.append((chunk, 3))  # High priority for title matches
                continue
            
            # Check summary
            if chunk.summary and query_lower in chunk.summary.lower():
                results.append((chunk, 2))  # Medium priority for summary matches
                continue
            
            # Check content
            if chunk.content and query_lower in chunk.content.lower():
                results.append((chunk, 1))  # Low priority for content matches
        
        # Sort by priority and hierarchical_id
        results.sort(key=lambda x: (-x[1], x[0].hierarchical_id))
        
        return [chunk for chunk, _ in results[:max_results]]
    
    def search_chunks_combined(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Combined semantic + text search"""
        # Get semantic results
        semantic_results = self.search_chunks_semantic(query, max_results//2)
        
        # Get text results
        text_chunks = self.search_chunks_text(query, max_results//2)
        text_results = [SearchResult(chunk=chunk, similarity_score=0.8, match_type="text") 
                       for chunk in text_chunks]
        
        # Combine and deduplicate
        all_results = semantic_results + text_results
        seen_ids = set()
        unique_results = []
        
        for result in all_results:
            if result.chunk.id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.chunk.id)
        
        # Sort by similarity score
        unique_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return unique_results[:max_results]
    
    def get_contextual_chunks(self, query: str, max_chunks: int = 10, 
                            threshold: float = 0.7) -> List[Dict]:
        """Get chunks similar to query for LLM context"""
        return self.embedder.get_contextual_chunks_for_query(
            query, list(self.chunks.values()), max_chunks, threshold
        )
    
    # ==================== PROCESSING OPERATIONS ====================
    
    def get_processing_context(self, chunk: ChunkData, all_chunks: List[ChunkData] = None) -> Dict[str, Any]:
        """Get context for LLM processing of chunk"""
        if all_chunks is None:
            all_chunks = list(self.chunks.values())
        
        all_chunks.sort(key=lambda x: x.hierarchical_id)
        
        # Find chunk position
        chunk_index = None
        for i, c in enumerate(all_chunks):
            if c.hierarchical_id == chunk.hierarchical_id:
                chunk_index = i
                break
        
        if chunk_index is None:
            return {'error': 'Chunk not found in sequence'}
        
        # Get context
        previous_chunk = all_chunks[chunk_index - 1] if chunk_index > 0 else None
        next_chunk = all_chunks[chunk_index + 1] if chunk_index < len(all_chunks) - 1 else None
        
        # Get 5 previous and next summaries
        start_prev = max(0, chunk_index - 5)
        end_next = min(len(all_chunks), chunk_index + 6)
        
        previous_summaries = [c.summary for c in all_chunks[start_prev:chunk_index] if c.summary]
        next_titles = [f"{c.hierarchical_id}: {c.title}" for c in all_chunks[chunk_index + 1:end_next]]
        
        return {
            'chunk': chunk,
            'previous_chunk': previous_chunk,
            'next_chunk': next_chunk,
            'previous_summaries': previous_summaries,
            'next_titles': next_titles,
            'position': f"{chunk_index + 1}/{len(all_chunks)}"
        }
    
    # ==================== STATISTICS & MANAGEMENT ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        storage_stats = self.persistence.get_storage_stats()
        faiss_stats = self.faiss_manager.get_index_stats()
        embedding_stats = self.embedder.get_embedding_stats()
        
        # Count by status
        status_counts = {}
        for chunk in self.chunks.values():
            status = chunk.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_chunks': len(self.chunks),
            'status_counts': status_counts,
            'storage': storage_stats,
            'faiss': faiss_stats,
            'embeddings': embedding_stats,
            'service_info': {
                'storage_dir': str(self.storage_dir),
                'embedding_model': self.embedding_model
            }
        }
    
    def cleanup_all(self) -> Dict[str, int]:
        """Cleanup orphaned data across all systems"""
        valid_chunk_ids = set(self.chunks.keys())
        
        results = {
            'orphaned_files': self.persistence.cleanup_orphaned_files(),
            'orphaned_embeddings': self.faiss_manager.cleanup_orphaned_embeddings(valid_chunk_ids)
        }
        
        return results
    
    def _generate_chunk_id(self) -> str:
        """Generate unique 8-character chunk ID"""
        return uuid.uuid4().hex[:8]
    
    def _extract_parent_id(self, hierarchical_id: str) -> Optional[str]:
        """Extract parent hierarchical ID"""
        parts = hierarchical_id.split('.')
        if len(parts) <= 1:
            return None
        return '.'.join(parts[:-1])
    
    def close(self):
        """Clean shutdown of service"""
        self.faiss_manager.save_index()
        print("📚 WriterService closed")