"""
Disk cache for OpenAI embeddings to avoid repeated API calls
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbeddingsCache:
   """Disk-based cache for OpenAI embeddings with metadata tracking"""
   
   def __init__(self, cache_dir: str = ".cache/embeddings", model: str = "text-embedding-3-small"):
       # Initialize cache directory structure
       self.cache_dir = Path(cache_dir)
       self.model = model
       self.model_dir = self.cache_dir / model
       self.model_dir.mkdir(parents=True, exist_ok=True)
       
       self.metadata_file = self.model_dir / "metadata.json"
       self._load_metadata()
       
       print(f"💾 Cache initialized: {self.model_dir} ({self.stats['total_cached']} cached)")
   
   def get(self, text_hash: str) -> Optional[np.ndarray]:
       # Load embedding from cache if exists
       cache_file = self.model_dir / f"{text_hash}.json"
       
       if not cache_file.exists():
           return None
       
       try:
           with open(cache_file, 'r', encoding='utf-8') as f:
               data = json.load(f)
           
           embedding = np.array(data['embedding'], dtype=np.float32)
           
           # Update access stats
           self.stats['cache_hits'] += 1
           print(f"💾 Cache HIT: {text_hash[:12]}...")
           
           return embedding
           
       except Exception as e:
           print(f"⚠️ Cache read error: {e}")
           return None
   
   def put(self, text_hash: str, embedding: np.ndarray, text_preview: str = "") -> bool:
       # Save embedding to cache with metadata
       cache_file = self.model_dir / f"{text_hash}.json"
       
       try:
           data = {
               'embedding': embedding.tolist(),
               'model': self.model,
               'text_preview': text_preview[:100],
               'cached_at': self._get_timestamp()
           }
           
           with open(cache_file, 'w', encoding='utf-8') as f:
               json.dump(data, f, indent=2)
           
           # Update stats
           self.stats['total_cached'] += 1
           self.stats['cache_misses'] += 1
           self._save_metadata()
           
           print(f"💾 Cache SAVE: {text_hash[:12]}... ({text_preview[:30]}...)")
           return True
           
       except Exception as e:
           print(f"❌ Cache write error: {e}")
           return False
   
   def get_batch(self, text_hashes: List[str]) -> Dict[str, Optional[np.ndarray]]:
       # Batch load embeddings from cache
       print(f"💾 Batch cache lookup: {len(text_hashes)} items")
       
       results = {}
       hits = 0
       
       for text_hash in text_hashes:
           embedding = self.get(text_hash)
           results[text_hash] = embedding
           if embedding is not None:
               hits += 1
       
       hit_rate = hits / len(text_hashes) if text_hashes else 0
       print(f"💾 Batch cache results: {hits}/{len(text_hashes)} hits ({hit_rate:.1%})")
       
       return results
   
   def put_batch(self, embeddings_data: Dict[str, tuple]) -> int:
       # Batch save embeddings to cache (hash -> (embedding, text_preview))
       print(f"💾 Batch cache save: {len(embeddings_data)} items")
       
       saved = 0
       for text_hash, (embedding, text_preview) in embeddings_data.items():
           if self.put(text_hash, embedding, text_preview):
               saved += 1
       
       print(f"💾 Batch save complete: {saved}/{len(embeddings_data)} saved")
       return saved
   
   def clear_cache(self) -> bool:
       # Remove all cached embeddings for this model
       try:
           import shutil
           if self.model_dir.exists():
               shutil.rmtree(self.model_dir)
               self.model_dir.mkdir(parents=True, exist_ok=True)
           
           self.stats = self._default_stats()
           self._save_metadata()
           
           print(f"🗑️ Cache cleared for model: {self.model}")
           return True
           
       except Exception as e:
           print(f"❌ Cache clear failed: {e}")
           return False
   
   def get_cache_stats(self) -> Dict:
       # Return cache usage statistics
       cache_size_mb = self._calculate_cache_size()
       
       return {
           **self.stats,
           'model': self.model,
           'cache_dir': str(self.model_dir),
           'cache_size_mb': cache_size_mb,
           'hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
       }
   
   def _load_metadata(self):
       # Load cache metadata from disk
       if self.metadata_file.exists():
           try:
               with open(self.metadata_file, 'r', encoding='utf-8') as f:
                   data = json.load(f)
               self.stats = data.get('stats', self._default_stats())
           except Exception:
               self.stats = self._default_stats()
       else:
           self.stats = self._default_stats()
   
   def _save_metadata(self):
       # Save cache metadata to disk
       try:
           metadata = {
               'model': self.model,
               'stats': self.stats,
               'last_updated': self._get_timestamp()
           }
           
           with open(self.metadata_file, 'w', encoding='utf-8') as f:
               json.dump(metadata, f, indent=2)
               
       except Exception as e:
           print(f"⚠️ Metadata save failed: {e}")
   
   def _default_stats(self) -> Dict:
       # Default statistics structure
       return {
           'total_cached': 0,
           'cache_hits': 0,
           'cache_misses': 0
       }
   
   def _calculate_cache_size(self) -> float:
       # Calculate total cache size in MB
       total_size = 0
       try:
           for cache_file in self.model_dir.glob("*.json"):
               total_size += cache_file.stat().st_size
           return total_size / (1024 * 1024)
       except Exception:
           return 0.0
   
   def _get_timestamp(self) -> str:
       # Get current timestamp string
       from datetime import datetime
       return datetime.now().isoformat()