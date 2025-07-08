"""
OpenAI Embeddings Client with batching and error handling
"""

import os
import hashlib
import logging
from typing import List, Union, Dict, Any
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIEmbeddingsClient:
    """
    OpenAI embeddings client with batching support
    Uses text-embedding-3-small model (1536 dimensions)
    """
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_dim = 1536  # text-embedding-3-small dimensions
        
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not found in environment")
        
        print(f"🔌 OpenAI Embeddings initialized: {model} ({self.embedding_dim}D)")
        logger.info(f"🔌 OpenAI Embeddings initialized: {model} ({self.embedding_dim}D)")
    
    def embed_single(self, text: str) -> np.ndarray:
        if not text.strip():
            print(f"⚪ Empty text, returning zero embedding")
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        text_preview = text[:50] + "..." if len(text) > 50 else text
        print(f"🧠 Generating embedding for: '{text_preview}'")
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text.strip(),
                encoding_format="float"
            )
            
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            print(f"✅ Embedding generated successfully (norm: {norm:.3f})")
            return embedding
            
        except Exception as e:
            print(f"❌ OpenAI embeddings API failed: {e}")
            logger.error(f"❌ OpenAI embeddings API failed: {e}")
            raise RuntimeError(f"Sorry, OpenAI embeddings service is down: {e}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 10) -> List[np.ndarray]:
        """PRAWDZIWY batching - każda grupa idzie osobno do OpenAI"""
        if not texts:
            print(f"⚪ No texts to embed")
            return []
        
        print(f"🚀 TRUE batching: {len(texts)} texts, {batch_size} per batch")
        
        all_embeddings = []
        total_batches = (len(texts) - 1) // batch_size + 1
        
        # Podziel na małe batche po 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"📦 Batch {batch_num}/{total_batches}: {len(batch)} texts")
            
            # Filtruj puste teksty
            non_empty_texts = [text.strip() for text in batch if text.strip()]
            
            if not non_empty_texts:
                print(f"⚪ All texts in batch {batch_num} are empty, using zero embeddings")
                batch_embeddings = [np.zeros(self.embedding_dim, dtype=np.float32) for _ in batch]
            else:
                try:
                    # PRAWDZIWY batch - tylko te teksty idą do OpenAI
                    print(f"🧠 Calling OpenAI API for {len(non_empty_texts)} texts...")
                    
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=non_empty_texts,
                        encoding_format="float"
                    )
                    
                    print(f"📡 OpenAI API response received")
                    
                    # Handle empty texts mapping
                    batch_embeddings = []
                    non_empty_idx = 0
                    
                    for text in batch:
                        if text.strip():
                            embedding = np.array(response.data[non_empty_idx].embedding, dtype=np.float32)
                            # Normalize
                            norm = np.linalg.norm(embedding)
                            if norm > 0:
                                embedding = embedding / norm
                            batch_embeddings.append(embedding)
                            non_empty_idx += 1
                        else:
                            batch_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                    
                    print(f"✅ Batch {batch_num} processed successfully")
                    
                except Exception as e:
                    print(f"❌ Batch {batch_num} failed: {e}")
                    logger.error(f"❌ Batch {batch_num} failed: {e}")
                    raise RuntimeError(f"OpenAI embeddings batch failed: {e}")
            
            all_embeddings.extend(batch_embeddings)
            
            if total_batches > 1:
                print(f"📊 Progress: {len(all_embeddings)}/{len(texts)} embeddings generated")
        
        print(f"🎉 TRUE batch embedding complete: {len(all_embeddings)} embeddings generated")
        return all_embeddings
    
    def embed_batch_with_cache(self, texts: List[str], cache, batch_size: int = 10) -> List[np.ndarray]:
        """Batching z sprawdzaniem cache - tylko brakujące teksty idą do OpenAI"""
        if not texts:
            return []
        
        print(f"🔍 Checking cache for {len(texts)} texts...")
        
        # 1. Sprawdź cache dla wszystkich tekstów
        text_hashes = [self.get_text_hash(text) for text in texts]
        cached_results = cache.get_batch(text_hashes)
        
        # 2. Znajdź teksty które trzeba wygenerować
        texts_to_generate = []
        texts_to_generate_hashes = []
        hash_to_index = {}
        
        for i, (text, text_hash) in enumerate(zip(texts, text_hashes)):
            if cached_results[text_hash] is None:
                hash_to_index[text_hash] = len(texts_to_generate)
                texts_to_generate.append(text)
                texts_to_generate_hashes.append(text_hash)
        
        print(f"💾 Cache: {len(texts) - len(texts_to_generate)} hits, {len(texts_to_generate)} misses")
        
        # 3. Wygeneruj brakujące embeddings (batching po 10 + cache po każdym batchu)
        new_embeddings = []
        if texts_to_generate:
            total_batches = (len(texts_to_generate) - 1) // batch_size + 1
            
            for i in range(0, len(texts_to_generate), batch_size):
                batch_texts = texts_to_generate[i:i + batch_size]
                batch_hashes = texts_to_generate_hashes[i:i + batch_size]
                batch_num = i // batch_size + 1
                
                print(f"📦 Processing batch {batch_num}/{total_batches}: {len(batch_texts)} texts")
                
                # Wygeneruj embeddings dla tego batcha
                batch_embeddings = self.embed_batch(batch_texts, batch_size=len(batch_texts))
                new_embeddings.extend(batch_embeddings)
                
                # ZAPISZ CACHE PO KAŻDYM BATCHU
                cache_data = {}
                for j, (text, text_hash) in enumerate(zip(batch_texts, batch_hashes)):
                    text_preview = text[:50] + "..." if len(text) > 50 else text
                    cache_data[text_hash] = (batch_embeddings[j], text_preview)
                
                cache.put_batch(cache_data)
                print(f"💾 Cache saved for batch {batch_num} ({len(cache_data)} embeddings)")
        
        # 4. Połącz wyniki: cache + nowe
        final_embeddings = []
        new_embedding_index = 0
        
        for text, text_hash in zip(texts, text_hashes):
            cached = cached_results[text_hash]
            if cached is not None:
                final_embeddings.append(cached)
            else:
                final_embeddings.append(new_embeddings[new_embedding_index])
                new_embedding_index += 1
        
        print(f"🎉 Batch with cache complete: {len(final_embeddings)} embeddings returned")
        return final_embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        if embedding1 is None or embedding2 is None:
            print(f"⚠️ Cannot compute similarity: one embedding is None")
            return 0.0
        
        # Since embeddings are normalized, dot product = cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        # Clamp to [0, 1] range
        similarity_clamped = max(0.0, min(1.0, float(similarity)))
        
        if similarity != similarity_clamped:
            print(f"⚠️ Similarity clamped: {similarity:.3f} → {similarity_clamped:.3f}")
        
        return similarity_clamped
    
    def get_text_hash(self, text: str) -> str:
        """Generate cache key for text"""
        hash_value = hashlib.sha256(text.encode('utf-8')).hexdigest()
        text_preview = text[:30] + "..." if len(text) > 30 else text
        print(f"🔑 Generated hash for '{text_preview}': {hash_value[:12]}...")
        return hash_value
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            "model": self.model,
            "embedding_dim": self.embedding_dim,
            "provider": "openai",
            "max_batch_size": 10
        }
        print(f"ℹ️ Model info: {info}")
        return info