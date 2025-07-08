import hashlib
import time
from typing import List, Optional, Dict, Tuple
import numpy as np
from pathlib import Path

from llm import OpenAIEmbeddingsClient, EmbeddingsCache
from .models import ChunkData, EmbeddingCache


class ChunkEmbedder:
    """Embeddings manager for chunks with caching - adapted from book-app pattern"""
    
    def __init__(self, model: str = "text-embedding-3-small", cache_dir: str = ".cache/embeddings"):
        self.model = model
        self.cache_dir = Path(cache_dir)
        
        # Initialize OpenAI client and cache
        self.embeddings_client = OpenAIEmbeddingsClient(model=model)
        self.cache = EmbeddingsCache(cache_dir=str(cache_dir), model=model)
        
        self.embedding_dim = 1536  # text-embedding-3-small dimensions
        
        print(f"🧠 ChunkEmbedder initialized: {model}")
        print(f"💾 Cache: {cache_dir}")
    
    def embed_chunk(self, chunk: ChunkData, use_content: bool = True) -> Optional[np.ndarray]:
        """Generate embedding for single chunk"""
        try:
            # Get text to embed
            if use_content and chunk.content:
                text = chunk.get_embedding_text()  # hierarchical_id + title + content
            else:
                text = f"{chunk.hierarchical_id} {chunk.title}"
            
            if not text.strip():
                print(f"⚠️ Empty text for chunk {chunk.id}")
                return np.zeros(self.embedding_dim, dtype=np.float32)
            
            # Check cache first
            text_hash = self._get_text_hash(text)
            cached_embedding = self.cache.get(text_hash)
            
            if cached_embedding is not None:
                print(f"💾 Cache hit for chunk {chunk.hierarchical_id}")
                return cached_embedding
            
            # Generate new embedding
            print(f"🧠 Generating embedding for chunk {chunk.hierarchical_id}")
            embedding = self.embeddings_client.embed_single(text)
            
            # Cache the result
            text_preview = f"{chunk.hierarchical_id}: {chunk.title}"
            self.cache.put(text_hash, embedding, text_preview)
            
            return embedding
            
        except Exception as e:
            print(f"❌ Failed to embed chunk {chunk.id}: {e}")
            return None
    
    def embed_chunks_batch(self, chunks: List[ChunkData], use_content: bool = True, 
                          batch_size: int = 10) -> List[Tuple[ChunkData, Optional[np.ndarray]]]:
        """Generate embeddings for multiple chunks with batching"""
        print(f"🚀 Batch embedding {len(chunks)} chunks (batch_size={batch_size})")
        
        results = []
        
        # Prepare texts and check cache
        texts_to_generate = []
        text_to_chunk_map = {}
        cached_results = {}
        
        for chunk in chunks:
            # Get text to embed
            if use_content and chunk.content:
                text = chunk.get_embedding_text()
            else:
                text = f"{chunk.hierarchical_id} {chunk.title}"
            
            if not text.strip():
                results.append((chunk, np.zeros(self.embedding_dim, dtype=np.float32)))
                continue
            
            # Check cache
            text_hash = self._get_text_hash(text)
            cached_embedding = self.cache.get(text_hash)
            
            if cached_embedding is not None:
                cached_results[chunk.id] = cached_embedding
                print(f"💾 Cache hit: {chunk.hierarchical_id}")
            else:
                texts_to_generate.append(text)
                text_to_chunk_map[text] = chunk
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if texts_to_generate:
            print(f"🧠 Generating {len(texts_to_generate)} new embeddings...")
            
            try:
                # Use batch embedding with cache
                new_embeddings = self.embeddings_client.embed_batch_with_cache(
                    texts_to_generate, 
                    self.cache, 
                    batch_size=batch_size
                )
                
            except Exception as e:
                print(f"❌ Batch embedding failed: {e}")
                # Fallback to individual embeddings
                new_embeddings = []
                for text in texts_to_generate:
                    try:
                        embedding = self.embeddings_client.embed_single(text)
                        new_embeddings.append(embedding)
                        
                        # Cache individual result
                        text_hash = self._get_text_hash(text)
                        chunk = text_to_chunk_map[text]
                        text_preview = f"{chunk.hierarchical_id}: {chunk.title}"
                        self.cache.put(text_hash, embedding, text_preview)
                        
                    except Exception as inner_e:
                        print(f"❌ Individual embedding failed: {inner_e}")
                        new_embeddings.append(None)
        
        # Combine results
        new_embedding_idx = 0
        for chunk in chunks:
            if chunk.id in cached_results:
                # Use cached result
                results.append((chunk, cached_results[chunk.id]))
            else:
                # Use new embedding
                if new_embedding_idx < len(new_embeddings):
                    embedding = new_embeddings[new_embedding_idx]
                    results.append((chunk, embedding))
                    new_embedding_idx += 1
                else:
                    # Fallback to zero embedding
                    results.append((chunk, np.zeros(self.embedding_dim, dtype=np.float32)))
        
        success_count = sum(1 for _, emb in results if emb is not None and not np.allclose(emb, 0))
        print(f"✅ Batch embedding complete: {success_count}/{len(chunks)} successful")
        
        return results
    
    def embed_text_query(self, query: str) -> Optional[np.ndarray]:
        """Generate embedding for search query"""
        if not query.strip():
            return None
        
        try:
            # Check cache
            text_hash = self._get_text_hash(query)
            cached_embedding = self.cache.get(text_hash)
            
            if cached_embedding is not None:
                print(f"💾 Cache hit for query")
                return cached_embedding
            
            # Generate new embedding
            print(f"🔍 Generating embedding for query: '{query[:50]}...'")
            embedding = self.embeddings_client.embed_single(query)
            
            # Cache the result
            self.cache.put(text_hash, embedding, query[:100])
            
            return embedding
            
        except Exception as e:
            print(f"❌ Failed to embed query: {e}")
            return None
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        return self.embeddings_client.compute_similarity(embedding1, embedding2)
    
    def get_contextual_chunks_for_query(self, query: str, all_chunks: List[ChunkData], 
                                       max_chunks: int = 10, threshold: float = 0.7) -> List[Dict]:
        """Get chunks similar to query for context (similar to NER contextual entities)"""
        query_embedding = self.embed_text_query(query)
        if query_embedding is None:
            return []
        
        similar_chunks = []
        
        for chunk in all_chunks:
            if chunk.embedding is None:
                continue
            
            similarity = self.compute_similarity(query_embedding, chunk.embedding)
            
            if similarity >= threshold:
                similar_chunks.append({
                    'chunk': chunk,
                    'similarity': similarity,
                    'hierarchical_id': chunk.hierarchical_id,
                    'title': chunk.title,
                    'summary': chunk.summary or chunk.title
                })
        
        # Sort by similarity and limit
        similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_chunks[:max_chunks]
    
    def update_chunk_embeddings(self, chunks: List[ChunkData], force_regenerate: bool = False) -> int:
        """Update embeddings for chunks that need them"""
        chunks_to_embed = []
        
        for chunk in chunks:
            needs_embedding = (
                chunk.embedding is None or 
                force_regenerate or
                (chunk.content and not hasattr(chunk, '_content_embedded'))
            )
            
            if needs_embedding:
                chunks_to_embed.append(chunk)
        
        if not chunks_to_embed:
            print("✅ All chunks already have embeddings")
            return 0
        
        print(f"🔄 Updating embeddings for {len(chunks_to_embed)} chunks")
        
        # Generate embeddings in batches
        results = self.embed_chunks_batch(chunks_to_embed, use_content=True)
        
        updated_count = 0
        for chunk, embedding in results:
            if embedding is not None and not np.allclose(embedding, 0):
                chunk.embedding = embedding
                chunk._content_embedded = True  # Mark as embedded
                updated_count += 1
        
        print(f"✅ Updated embeddings for {updated_count} chunks")
        return updated_count
    
    def get_embedding_stats(self) -> Dict:
        """Get embedding statistics"""
        cache_stats = self.cache.get_cache_stats()
        
        return {
            'model': self.model,
            'embedding_dim': self.embedding_dim,
            'cache_stats': cache_stats,
            'embedder_info': self.embeddings_client.get_model_info()
        }
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        return self.embeddings_client.get_text_hash(text)
    
    def _get_cached_embedding(self, text: str, cache_key: str) -> Optional[np.ndarray]:
        """Get cached embedding with fallback key"""
        text_hash = self._get_text_hash(text)
        
        # Try main hash first
        embedding = self.cache.get(text_hash)
        if embedding is not None:
            return embedding
        
        # Try fallback key if provided
        if cache_key != text_hash:
            return self.cache.get(cache_key)
        
        return None
    
    def clear_cache(self) -> bool:
        """Clear embeddings cache"""
        return self.cache.clear_cache()
    
    def precompute_embeddings_for_toc(self, toc_entries: List[str]) -> Dict[str, np.ndarray]:
        """Precompute embeddings for TOC entries (for initial setup)"""
        print(f"📝 Precomputing embeddings for {len(toc_entries)} TOC entries")
        
        embeddings = {}
        batch_results = self.embeddings_client.embed_batch_with_cache(
            toc_entries, 
            self.cache,
            batch_size=10
        )
        
        for text, embedding in zip(toc_entries, batch_results):
            if embedding is not None:
                embeddings[text] = embedding
        
        print(f"✅ Precomputed {len(embeddings)} TOC embeddings")
        return embeddings