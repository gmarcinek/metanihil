"""
Search endpoints - semantic and text search
"""

from fastapi import APIRouter, HTTPException
from api.main import get_writer_service
from api.models import SearchRequest, SearchResponse, SearchResult, ChunkResponse
from writer.models import ChunkData


router = APIRouter(prefix="/search", tags=["search"])


def _chunk_to_response(chunk: ChunkData) -> ChunkResponse:
    """Convert ChunkData to ChunkResponse"""
    return ChunkResponse(
        id=chunk.id,
        hierarchical_id=chunk.hierarchical_id,
        parent_hierarchical_id=chunk.parent_hierarchical_id,
        title=chunk.title,
        level=chunk.level,
        display_level=chunk.display_level,
        status=chunk.status.value,
        content=chunk.content,
        summary=chunk.summary,
        embedding_model=chunk.embedding_model,
        created_at=chunk.created_at,
        updated_at=chunk.updated_at,
        has_embedding=chunk.embedding is not None
    )


@router.post("", response_model=SearchResponse)
async def search_chunks(request: SearchRequest):
    """Search chunks - semantic, text, or combined"""
    try:
        service = get_writer_service()
        
        if request.search_type == "semantic":
            results = service.search_chunks_semantic(
                query=request.query,
                max_results=request.max_results,
                min_similarity=request.min_similarity
            )
            search_results = [
                SearchResult(
                    chunk=_chunk_to_response(r.chunk),
                    similarity_score=r.similarity_score,
                    match_type=r.match_type
                ) for r in results
            ]
            
        elif request.search_type == "text":
            chunks = service.search_chunks_text(
                query=request.query,
                max_results=request.max_results
            )
            search_results = [
                SearchResult(
                    chunk=_chunk_to_response(chunk),
                    similarity_score=0.8,  # Default for text matches
                    match_type="text"
                ) for chunk in chunks
            ]
            
        else:  # combined
            results = service.search_chunks_combined(
                query=request.query,
                max_results=request.max_results
            )
            search_results = [
                SearchResult(
                    chunk=_chunk_to_response(r.chunk),
                    similarity_score=r.similarity_score,
                    match_type=r.match_type
                ) for r in results
            ]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_found=len(search_results),
            search_type=request.search_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/contextual")
async def get_contextual_chunks(
    query: str,
    max_chunks: int = 10,
    threshold: float = 0.7
):
    """Get contextual chunks for LLM processing"""
    try:
        service = get_writer_service()
        
        contextual_chunks = service.get_contextual_chunks(
            query=query,
            max_chunks=max_chunks,
            threshold=threshold
        )
        
        return {
            "query": query,
            "chunks": contextual_chunks,
            "total_found": len(contextual_chunks)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Contextual search failed: {str(e)}")