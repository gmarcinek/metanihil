"""
Pydantic models for WriterService API endpoints
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ChunkStatusEnum(str, Enum):
    """Chunk status for API responses"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkResponse(BaseModel):
    """Chunk data for API responses"""
    id: str
    hierarchical_id: str
    parent_hierarchical_id: Optional[str] = None
    title: str
    level: int
    display_level: int
    status: ChunkStatusEnum
    content: Optional[str] = None
    summary: Optional[str] = None
    embedding_model: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    has_embedding: bool = False


class ChunkCreateRequest(BaseModel):
    """Request to create new chunk"""
    hierarchical_id: str = Field(..., description="Hierarchical ID like 1.2.3")
    title: str = Field(..., min_length=1, max_length=500)
    parent_hierarchical_id: Optional[str] = None
    content: Optional[str] = Field(None, max_length=50000)
    summary: Optional[str] = Field(None, max_length=2000)


class ChunkUpdateRequest(BaseModel):
    """Request to update chunk"""
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    content: Optional[str] = Field(None, max_length=50000)
    summary: Optional[str] = Field(None, max_length=2000)
    status: Optional[ChunkStatusEnum] = None


class SearchRequest(BaseModel):
    """Search request"""
    query: str = Field(..., min_length=1, max_length=1000)
    max_results: int = Field(10, ge=1, le=50)
    min_similarity: float = Field(0.7, ge=0.0, le=1.0)
    search_type: str = Field("combined", regex="^(semantic|text|combined)$")


class SearchResult(BaseModel):
    """Single search result"""
    chunk: ChunkResponse
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    match_type: str


class SearchResponse(BaseModel):
    """Search results response"""
    query: str
    results: List[SearchResult]
    total_found: int
    search_type: str


class TOCProcessRequest(BaseModel):
    """Request to process TOC content"""
    toc_content: str = Field(..., min_length=1)
    auto_embed: bool = Field(True, description="Automatically generate embeddings")


class TOCProcessResponse(BaseModel):
    """Response from TOC processing"""
    status: str
    chunks_created: int
    chunks_embedded: int = 0
    message: str


class ProcessingStatsResponse(BaseModel):
    """Processing statistics"""
    total_chunks: int
    processed_chunks: int
    failed_chunks: int
    skipped_chunks: int
    processing_time_seconds: float = 0.0
    success_rate: float


class ContextualChunksRequest(BaseModel):
    """Request for contextual chunks"""
    query: str = Field(..., min_length=1)
    max_chunks: int = Field(10, ge=1, le=20)
    threshold: float = Field(0.7, ge=0.0, le=1.0)


class ContextualChunk(BaseModel):
    """Contextual chunk for LLM context"""
    hierarchical_id: str
    title: str
    summary: str
    similarity: float


class ContextualChunksResponse(BaseModel):
    """Response with contextual chunks"""
    query: str
    chunks: List[ContextualChunk]
    total_found: int


class ServiceStatsResponse(BaseModel):
    """WriterService statistics"""
    total_chunks: int
    status_counts: Dict[str, int]
    storage_size_mb: float
    faiss_vectors: int
    embedding_model: str
    cache_stats: Dict[str, Any]


class ChunkListResponse(BaseModel):
    """Paginated chunk list"""
    chunks: List[ChunkResponse]
    total_chunks: int
    page: int
    per_page: int
    has_next: bool


class ProcessingContextResponse(BaseModel):
    """Processing context for chunk"""
    chunk: ChunkResponse
    previous_chunk: Optional[ChunkResponse] = None
    next_chunk: Optional[ChunkResponse] = None
    previous_summaries: List[str] = []
    next_titles: List[str] = []
    position: str


class EmbeddingRequest(BaseModel):
    """Request to generate embeddings"""
    chunk_ids: Optional[List[str]] = None
    force_regenerate: bool = False
    use_content: bool = True


class EmbeddingResponse(BaseModel):
    """Response from embedding generation"""
    status: str
    chunks_processed: int
    chunks_embedded: int
    message: str


class ErrorResponse(BaseModel):
    """API error response"""
    error: str
    detail: Optional[str] = None
    chunk_id: Optional[str] = None


# Response wrapper models
class SuccessResponse(BaseModel):
    """Generic success response"""
    status: str = "success"
    message: str
    data: Optional[Dict[str, Any]] = None


class ChunkOperationResponse(BaseModel):
    """Response from chunk operations"""
    status: str
    chunk_id: str
    message: str
    chunk: Optional[ChunkResponse] = None