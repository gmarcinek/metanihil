from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np


class ChunkStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ChunkData:
    """Core chunk model - używany przez pipeline i API"""
    id: str
    hierarchical_id: str
    parent_hierarchical_id: Optional[str]
    title: str
    level: int
    display_level: int
    status: ChunkStatus
    content: Optional[str] = None
    summary: Optional[str] = None
    
    # Nowe pola dla embeddings i API
    embedding: Optional[np.ndarray] = None
    embedding_model: Optional[str] = "text-embedding-3-small"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        """Auto-set timestamps if not provided"""
        now = datetime.now().isoformat()
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now
    
    def get_embedding_text(self) -> str:
        """Get text for embedding generation"""
        if self.content:
            return f"{self.hierarchical_id} {self.title}\n{self.content}"
        else:
            return f"{self.hierarchical_id} {self.title}"
    
    def get_summary_text(self) -> str:
        """Get text for summary embedding"""
        return f"{self.hierarchical_id} {self.title}\n{self.summary or ''}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "hierarchical_id": self.hierarchical_id,
            "parent_hierarchical_id": self.parent_hierarchical_id,
            "title": self.title,
            "level": self.level,
            "display_level": self.display_level,
            "status": self.status.value,
            "content": self.content,
            "summary": self.summary,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "has_embedding": self.embedding is not None
        }
    
    def update_timestamp(self):
        """Update timestamp when chunk is modified"""
        self.updated_at = datetime.now().isoformat()


@dataclass
class ProcessingStats:
    """Statistics for processing operations"""
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    skipped_chunks: int = 0
    processing_time_seconds: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return (self.processed_chunks / self.total_chunks) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "failed_chunks": self.failed_chunks,
            "skipped_chunks": self.skipped_chunks,
            "processing_time_seconds": self.processing_time_seconds,
            "success_rate": round(self.success_rate, 2)
        }


@dataclass
class SearchResult:
    """Result from semantic search"""
    chunk: ChunkData
    similarity_score: float
    match_type: str = "content"  # "content", "summary", "title"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "similarity_score": round(self.similarity_score, 3),
            "match_type": self.match_type
        }


@dataclass 
class TOCEntry:
    """Entry from Table of Contents parsing"""
    hierarchical_id: str
    title: str
    parent_hierarchical_id: Optional[str] = None
    level: int = 1
    
    def to_chunk_data(self, chunk_id: str) -> ChunkData:
        """Convert TOC entry to ChunkData"""
        display_level = min(self.level, 3)
        
        return ChunkData(
            id=chunk_id,
            hierarchical_id=self.hierarchical_id,
            parent_hierarchical_id=self.parent_hierarchical_id,
            title=self.title,
            level=self.level,
            display_level=display_level,
            status=ChunkStatus.NOT_STARTED
        )


@dataclass
class EmbeddingCache:
    """Cache entry for embeddings"""
    text_hash: str
    embedding: np.ndarray
    model: str
    created_at: str
    text_preview: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_hash": self.text_hash,
            "model": self.model,
            "created_at": self.created_at,
            "text_preview": self.text_preview,
            "embedding_shape": self.embedding.shape if self.embedding is not None else None
        }


# Constants for validation
VALID_STATUSES = {status.value for status in ChunkStatus}
MAX_TITLE_LENGTH = 500
MAX_CONTENT_LENGTH = 50000
MAX_SUMMARY_LENGTH = 2000