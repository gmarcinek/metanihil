import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime

from .models import ChunkData, ChunkStatus


class ChunkPersistence:
    """JSON-based persistence for chunks - adapted from book-app pattern"""
    
    def __init__(self, storage_dir: str = "output/writer_storage"):
        self.storage_dir = Path(storage_dir)
        self.chunks_dir = self.storage_dir / "chunks"
        self.metadata_file = self.storage_dir / "metadata.json"
        
        # Create directories
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create metadata
        self.metadata = self._load_metadata()
        
        print(f"📁 Chunk persistence initialized: {self.storage_dir}")
        print(f"📊 Found {len(self.metadata.get('chunk_index', {}))} chunks")
    
    def save_chunk(self, chunk: ChunkData) -> bool:
        """Save single chunk to JSON file"""
        try:
            chunk.update_timestamp()
            
            # Prepare chunk data for JSON (exclude numpy arrays)
            chunk_data = chunk.to_dict()
            
            # Remove embedding from JSON (stored separately in FAISS)
            chunk_data.pop('has_embedding', None)
            
            # Save to individual JSON file
            chunk_file = self.chunks_dir / f"{chunk.id}.json"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)
            
            # Update metadata index
            self._update_chunk_index(chunk)
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to save chunk {chunk.id}: {e}")
            return False
    
    def save_chunks(self, chunks: List[ChunkData]) -> int:
        """Save multiple chunks"""
        saved_count = 0
        
        for chunk in chunks:
            if self.save_chunk(chunk):
                saved_count += 1
        
        # Save metadata after batch operation
        self._save_metadata()
        
        print(f"💾 Saved {saved_count}/{len(chunks)} chunks")
        return saved_count
    
    def load_chunk(self, chunk_id: str) -> Optional[ChunkData]:
        """Load single chunk by ID"""
        chunk_file = self.chunks_dir / f"{chunk_id}.json"
        
        if not chunk_file.exists():
            return None
        
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return self._dict_to_chunk(data)
            
        except Exception as e:
            print(f"❌ Failed to load chunk {chunk_id}: {e}")
            return None
    
    def load_all_chunks(self) -> Dict[str, ChunkData]:
        """Load all chunks from storage"""
        chunks = {}
        
        for chunk_id in self.metadata.get('chunk_index', {}):
            chunk = self.load_chunk(chunk_id)
            if chunk:
                chunks[chunk_id] = chunk
        
        print(f"📖 Loaded {len(chunks)} chunks from storage")
        return chunks
    
    def get_chunks_by_status(self, status: ChunkStatus) -> List[ChunkData]:
        """Get chunks filtered by status"""
        chunks = []
        
        for chunk_id in self.metadata.get('chunk_index', {}):
            chunk = self.load_chunk(chunk_id)
            if chunk and chunk.status == status:
                chunks.append(chunk)
        
        # Sort by hierarchical_id
        chunks.sort(key=lambda x: x.hierarchical_id)
        return chunks
    
    def get_chunks_by_hierarchical_pattern(self, pattern: str) -> List[ChunkData]:
        """Get chunks matching hierarchical pattern (e.g., '1.2.*')"""
        chunks = []
        
        for chunk_id in self.metadata.get('chunk_index', {}):
            chunk_meta = self.metadata['chunk_index'][chunk_id]
            hierarchical_id = chunk_meta.get('hierarchical_id', '')
            
            if self._matches_pattern(hierarchical_id, pattern):
                chunk = self.load_chunk(chunk_id)
                if chunk:
                    chunks.append(chunk)
        
        chunks.sort(key=lambda x: x.hierarchical_id)
        return chunks
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete chunk from storage"""
        try:
            # Remove file
            chunk_file = self.chunks_dir / f"{chunk_id}.json"
            if chunk_file.exists():
                chunk_file.unlink()
            
            # Remove from index
            if chunk_id in self.metadata.get('chunk_index', {}):
                del self.metadata['chunk_index'][chunk_id]
                self._save_metadata()
            
            print(f"🗑️ Deleted chunk {chunk_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to delete chunk {chunk_id}: {e}")
            return False
    
    def get_storage_stats(self) -> Dict[str, any]:
        """Get storage statistics"""
        chunk_index = self.metadata.get('chunk_index', {})
        
        # Count by status
        status_counts = {}
        total_size = 0
        
        for chunk_id, chunk_meta in chunk_index.items():
            status = chunk_meta.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Calculate file size
            chunk_file = self.chunks_dir / f"{chunk_id}.json"
            if chunk_file.exists():
                total_size += chunk_file.stat().st_size
        
        return {
            'total_chunks': len(chunk_index),
            'status_counts': status_counts,
            'storage_size_mb': round(total_size / (1024 * 1024), 2),
            'storage_dir': str(self.storage_dir),
            'last_updated': self.metadata.get('last_updated')
        }
    
    def cleanup_orphaned_files(self) -> int:
        """Remove JSON files not in index"""
        removed_count = 0
        indexed_ids = set(self.metadata.get('chunk_index', {}).keys())
        
        for json_file in self.chunks_dir.glob("*.json"):
            chunk_id = json_file.stem
            if chunk_id not in indexed_ids:
                json_file.unlink()
                removed_count += 1
        
        if removed_count > 0:
            print(f"🧹 Cleaned up {removed_count} orphaned files")
        
        return removed_count
    
    def _load_metadata(self) -> Dict:
        """Load metadata index"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load metadata: {e}")
        
        # Return default metadata structure
        return {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'chunk_index': {}
        }
    
    def _save_metadata(self):
        """Save metadata index"""
        self.metadata['last_updated'] = datetime.now().isoformat()
        
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ Failed to save metadata: {e}")
    
    def _update_chunk_index(self, chunk: ChunkData):
        """Update chunk in metadata index"""
        if 'chunk_index' not in self.metadata:
            self.metadata['chunk_index'] = {}
        
        self.metadata['chunk_index'][chunk.id] = {
            'hierarchical_id': chunk.hierarchical_id,
            'title': chunk.title,
            'status': chunk.status.value,
            'level': chunk.level,
            'updated_at': chunk.updated_at,
            'has_content': chunk.content is not None,
            'has_summary': chunk.summary is not None
        }
    
    def _dict_to_chunk(self, data: Dict) -> ChunkData:
        """Convert dictionary to ChunkData"""
        return ChunkData(
            id=data['id'],
            hierarchical_id=data['hierarchical_id'],
            parent_hierarchical_id=data.get('parent_hierarchical_id'),
            title=data['title'],
            level=data['level'],
            display_level=data['display_level'],
            status=ChunkStatus(data['status']),
            content=data.get('content'),
            summary=data.get('summary'),
            embedding=None,  # Embeddings stored in FAISS
            embedding_model=data.get('embedding_model', 'text-embedding-3-small'),
            created_at=data.get('created_at'),
            updated_at=data.get('updated_at')
        )
    
    def _matches_pattern(self, hierarchical_id: str, pattern: str) -> bool:
        """Check if hierarchical_id matches pattern (supports wildcards)"""
        if '*' not in pattern:
            return hierarchical_id == pattern
        
        # Simple wildcard matching
        pattern_parts = pattern.split('.')
        id_parts = hierarchical_id.split('.')
        
        for i, pattern_part in enumerate(pattern_parts):
            if pattern_part == '*':
                return True  # Wildcard matches rest
            
            if i >= len(id_parts) or pattern_part != id_parts[i]:
                return False
        
        return len(pattern_parts) == len(id_parts)