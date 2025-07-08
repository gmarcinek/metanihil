import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️ FAISS not available - semantic search disabled")

from .models import ChunkData, SearchResult

logger = logging.getLogger(__name__)


class FAISSManager:
    """FAISS index manager for chunk embeddings - adapted from book-app pattern"""
    
    def __init__(self, storage_dir: str = "data/writer_storage", embedding_dim: int = 1536):
        self.storage_dir = Path(storage_dir)
        self.faiss_dir = self.storage_dir / "faiss"
        self.embedding_dim = embedding_dim
        
        # FAISS files
        self.index_file = self.faiss_dir / "chunks.index"
        self.id_mapping_file = self.faiss_dir / "id_mapping.json"
        self.metadata_file = self.faiss_dir / "faiss_metadata.json"
        
        # Create directory
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS components
        self.index = None
        self.id_to_faiss_id = {}  # chunk_id -> faiss_internal_id
        self.faiss_id_to_id = {}  # faiss_internal_id -> chunk_id
        self.next_faiss_id = 0
        
        if FAISS_AVAILABLE:
            self._load_or_create_index()
        else:
            print("❌ FAISS not available - semantic search disabled")
    
    def _load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        if self._index_files_exist():
            self._load_index()
        else:
            self._create_new_index()
    
    def _index_files_exist(self) -> bool:
        """Check if all FAISS files exist"""
        return (self.index_file.exists() and 
                self.id_mapping_file.exists() and 
                self.metadata_file.exists())
    
    def _load_index(self):
        """Load existing FAISS index and mappings"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_file))
            
            # Load ID mappings
            with open(self.id_mapping_file, 'r') as f:
                mapping_data = json.load(f)
                self.id_to_faiss_id = mapping_data.get('id_to_faiss_id', {})
                self.faiss_id_to_id = mapping_data.get('faiss_id_to_id', {})
                # Convert string keys back to int for faiss_id_to_id
                self.faiss_id_to_id = {int(k): v for k, v in self.faiss_id_to_id.items()}
            
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                self.next_faiss_id = metadata.get('next_faiss_id', 0)
            
            print(f"📖 Loaded FAISS index: {self.index.ntotal} vectors")
            
        except Exception as e:
            print(f"❌ Failed to load FAISS index: {e}")
            self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index"""
        if not FAISS_AVAILABLE:
            return
        
        # Create flat index with inner product (for normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.id_to_faiss_id = {}
        self.faiss_id_to_id = {}
        self.next_faiss_id = 0
        
        print(f"🆕 Created new FAISS index (dim={self.embedding_dim})")
        self._save_index()
    
    def add_chunk_embedding(self, chunk: ChunkData, embedding: np.ndarray) -> bool:
        """Add or update chunk embedding in FAISS"""
        if not FAISS_AVAILABLE or self.index is None:
            return False
        
        try:
            # Remove existing embedding if exists
            if chunk.id in self.id_to_faiss_id:
                self.remove_chunk_embedding(chunk.id)
            
            # Normalize embedding
            embedding = embedding.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(embedding)
            
            # Add to FAISS index
            faiss_id = self.next_faiss_id
            self.index.add(embedding)
            
            # Update mappings
            self.id_to_faiss_id[chunk.id] = faiss_id
            self.faiss_id_to_id[faiss_id] = chunk.id
            self.next_faiss_id += 1
            
            print(f"✅ Added embedding for chunk {chunk.id} ({chunk.hierarchical_id})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to add embedding for chunk {chunk.id}: {e}")
            return False
    
    def remove_chunk_embedding(self, chunk_id: str) -> bool:
        """Remove chunk embedding from FAISS (requires rebuild)"""
        if not FAISS_AVAILABLE or chunk_id not in self.id_to_faiss_id:
            return False
        
        try:
            # Mark for removal and trigger rebuild
            faiss_id = self.id_to_faiss_id[chunk_id]
            del self.id_to_faiss_id[chunk_id]
            del self.faiss_id_to_id[faiss_id]
            
            # FAISS doesn't support removal, so we need to rebuild
            self._rebuild_index_from_mappings()
            
            print(f"🗑️ Removed embedding for chunk {chunk_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to remove embedding for chunk {chunk_id}: {e}")
            return False
    
    def search_similar_chunks(self, query_embedding: np.ndarray, max_results: int = 10, 
                             min_similarity: float = 0.7) -> List[Tuple[str, float]]:
        """Search for similar chunks using FAISS"""
        if not FAISS_AVAILABLE or self.index is None or self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS
            k = min(max_results, self.index.ntotal)
            similarities, faiss_ids = self.index.search(query_embedding, k)
            
            results = []
            for similarity, faiss_id in zip(similarities[0], faiss_ids[0]):
                if faiss_id == -1:  # No more results
                    break
                
                if similarity < min_similarity:
                    continue
                
                chunk_id = self.faiss_id_to_id.get(faiss_id)
                if chunk_id:
                    results.append((chunk_id, float(similarity)))
            
            return results
            
        except Exception as e:
            print(f"❌ FAISS search failed: {e}")
            return []
    
    def get_chunk_neighbors(self, chunk_id: str, max_neighbors: int = 5) -> List[Tuple[str, float]]:
        """Get semantically similar chunks to given chunk"""
        if chunk_id not in self.id_to_faiss_id:
            return []
        
        try:
            # Get the chunk's embedding vector
            faiss_id = self.id_to_faiss_id[chunk_id]
            
            # Reconstruct the embedding (only works with flat indexes)
            if hasattr(self.index, 'reconstruct'):
                embedding = self.index.reconstruct(faiss_id)
                return self.search_similar_chunks(embedding, max_neighbors + 1, min_similarity=0.5)
            else:
                print("⚠️ Index doesn't support reconstruction")
                return []
                
        except Exception as e:
            print(f"❌ Failed to get neighbors for chunk {chunk_id}: {e}")
            return []
    
    def rebuild_index_from_chunks(self, chunks_with_embeddings: List[Tuple[ChunkData, np.ndarray]]) -> bool:
        """Rebuild entire FAISS index from scratch"""
        if not FAISS_AVAILABLE:
            return False
        
        try:
            print(f"🔄 Rebuilding FAISS index with {len(chunks_with_embeddings)} chunks...")
            
            # Create new index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.id_to_faiss_id = {}
            self.faiss_id_to_id = {}
            self.next_faiss_id = 0
            
            # Add all embeddings
            for chunk, embedding in chunks_with_embeddings:
                self.add_chunk_embedding(chunk, embedding)
            
            # Save new index
            self._save_index()
            
            print(f"✅ FAISS index rebuilt with {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            print(f"❌ Failed to rebuild FAISS index: {e}")
            return False
    
    def _rebuild_index_from_mappings(self):
        """Rebuild index keeping only chunks still in mappings"""
        if not FAISS_AVAILABLE:
            return
        
        # Get current valid chunk IDs
        valid_chunk_ids = set(self.id_to_faiss_id.keys())
        
        if not valid_chunk_ids:
            self._create_new_index()
            return
        
        print(f"🔄 Rebuilding FAISS index (keeping {len(valid_chunk_ids)} chunks)")
        
        # This is a simplified rebuild - in practice, you'd need to reload embeddings
        # For now, just clean up mappings and create new index
        self._create_new_index()
    
    def save_index(self):
        """Save FAISS index and mappings to disk"""
        self._save_index()
    
    def _save_index(self):
        """Internal method to save FAISS index and mappings"""
        if not FAISS_AVAILABLE or self.index is None:
            return
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save ID mappings
            mapping_data = {
                'id_to_faiss_id': self.id_to_faiss_id,
                'faiss_id_to_id': {str(k): v for k, v in self.faiss_id_to_id.items()}
            }
            with open(self.id_mapping_file, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            
            # Save metadata
            metadata = {
                'version': '1.0',
                'embedding_dim': self.embedding_dim,
                'next_faiss_id': self.next_faiss_id,
                'total_vectors': self.index.ntotal,
                'index_type': 'IndexFlatIP'
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
        except Exception as e:
            print(f"❌ Failed to save FAISS index: {e}")
    
    def get_index_stats(self) -> Dict:
        """Get FAISS index statistics"""
        if not FAISS_AVAILABLE or self.index is None:
            return {'available': False}
        
        return {
            'available': True,
            'total_vectors': self.index.ntotal,
            'embedding_dim': self.embedding_dim,
            'index_type': type(self.index).__name__,
            'total_chunks_mapped': len(self.id_to_faiss_id),
            'next_faiss_id': self.next_faiss_id,
            'storage_dir': str(self.faiss_dir)
        }
    
    def chunk_has_embedding(self, chunk_id: str) -> bool:
        """Check if chunk has embedding in FAISS"""
        return chunk_id in self.id_to_faiss_id
    
    def get_all_chunk_ids_with_embeddings(self) -> Set[str]:
        """Get all chunk IDs that have embeddings"""
        return set(self.id_to_faiss_id.keys())
    
    def cleanup_orphaned_embeddings(self, valid_chunk_ids: Set[str]) -> int:
        """Remove embeddings for chunks that no longer exist"""
        orphaned_count = 0
        
        for chunk_id in list(self.id_to_faiss_id.keys()):
            if chunk_id not in valid_chunk_ids:
                self.remove_chunk_embedding(chunk_id)
                orphaned_count += 1
        
        if orphaned_count > 0:
            print(f"🧹 Cleaned up {orphaned_count} orphaned embeddings")
            self._save_index()
        
        return orphaned_count