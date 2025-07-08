from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from api.main import get_writer_service
from api.models import (
    ChunkResponse, ChunkCreateRequest, ChunkUpdateRequest,
    ChunkListResponse, ChunkOperationResponse, ProcessingContextResponse
)
from writer.models import ChunkStatus, ChunkData


router = APIRouter(prefix="/chunks", tags=["chunks"])


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


@router.get("", response_model=ChunkListResponse)
async def get_chunks(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    hierarchical_pattern: Optional[str] = Query(None, description="Pattern like '1.2.*'")
):
    """Get chunks with pagination and filtering"""
    try:
        service = get_writer_service()
        
        # Get chunks based on filters
        if status:
            try:
                chunk_status = ChunkStatus(status)
                all_chunks = service.get_chunks_by_status(chunk_status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        elif hierarchical_pattern:
            all_chunks = service.persistence.get_chunks_by_hierarchical_pattern(hierarchical_pattern)
        else:
            all_chunks = list(service.chunks.values())
        
        # Sort by hierarchical_id
        all_chunks.sort(key=lambda x: x.hierarchical_id)
        
        # Pagination
        total_chunks = len(all_chunks)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_chunks = all_chunks[start_idx:end_idx]
        
        return ChunkListResponse(
            chunks=[_chunk_to_response(chunk) for chunk in paginated_chunks],
            total_chunks=total_chunks,
            page=page,
            per_page=per_page,
            has_next=end_idx < total_chunks
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chunks: {str(e)}")


@router.post("", response_model=ChunkOperationResponse)
async def create_chunk(request: ChunkCreateRequest):
    """Create new chunk"""
    try:
        service = get_writer_service()
        
        # Check if hierarchical_id already exists
        existing = service.get_chunk_by_hierarchical_id(request.hierarchical_id)
        if existing:
            raise HTTPException(
                status_code=409, 
                detail=f"Chunk with hierarchical_id '{request.hierarchical_id}' already exists"
            )
        
        # Create new chunk
        from writer.models import TOCEntry
        import uuid
        
        # Determine level and parent
        level = len(request.hierarchical_id.split('.'))
        
        toc_entry = TOCEntry(
            hierarchical_id=request.hierarchical_id,
            title=request.title,
            parent_hierarchical_id=request.parent_hierarchical_id,
            level=level
        )
        
        chunk_id = uuid.uuid4().hex[:8]
        chunk = toc_entry.to_chunk_data(chunk_id)
        
        # Add optional content and summary
        if request.content:
            chunk.content = request.content
        if request.summary:
            chunk.summary = request.summary
        
        # Save chunk
        saved_count = service.save_toc_chunks([chunk])
        
        if saved_count == 1:
            return ChunkOperationResponse(
                status="created",
                chunk_id=chunk.id,
                message=f"Created chunk {request.hierarchical_id}",
                chunk=_chunk_to_response(chunk)
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save chunk")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chunk: {str(e)}")


@router.get("/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(chunk_id: str):
    """Get chunk by ID"""
    try:
        service = get_writer_service()
        chunk = service.get_chunk_by_id(chunk_id)
        
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        return _chunk_to_response(chunk)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chunk: {str(e)}")


@router.get("/hierarchical/{hierarchical_id}", response_model=ChunkResponse)
async def get_chunk_by_hierarchical_id(hierarchical_id: str):
    """Get chunk by hierarchical ID"""
    try:
        service = get_writer_service()
        chunk = service.get_chunk_by_hierarchical_id(hierarchical_id)
        
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk with hierarchical_id '{hierarchical_id}' not found")
        
        return _chunk_to_response(chunk)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chunk: {str(e)}")


@router.put("/{chunk_id}", response_model=ChunkOperationResponse)
async def update_chunk(chunk_id: str, request: ChunkUpdateRequest):
    """Update chunk"""
    try:
        service = get_writer_service()
        chunk = service.get_chunk_by_id(chunk_id)
        
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        # Update fields
        updated_fields = []
        
        if request.title is not None:
            chunk.title = request.title
            updated_fields.append("title")
        
        if request.status is not None:
            chunk.status = ChunkStatus(request.status)
            updated_fields.append("status")
        
        # Handle content and summary updates (triggers embedding regeneration)
        if request.content is not None or request.summary is not None:
            new_content = request.content if request.content is not None else chunk.content
            new_summary = request.summary if request.summary is not None else chunk.summary
            
            success = service.update_chunk_content(chunk_id, new_content, new_summary)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to update chunk content")
            
            if request.content is not None:
                updated_fields.append("content")
            if request.summary is not None:
                updated_fields.append("summary")
        else:
            # Just update metadata without triggering embedding regeneration
            chunk.update_timestamp()
            service.persistence.save_chunk(chunk)
            service.chunks[chunk_id] = chunk
        
        return ChunkOperationResponse(
            status="updated",
            chunk_id=chunk_id,
            message=f"Updated chunk {chunk_id}: {', '.join(updated_fields)}",
            chunk=_chunk_to_response(chunk)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update chunk: {str(e)}")


@router.delete("/{chunk_id}", response_model=ChunkOperationResponse)
async def delete_chunk(chunk_id: str):
    """Delete chunk"""
    try:
        service = get_writer_service()
        chunk = service.get_chunk_by_id(chunk_id)
        
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        success = service.delete_chunk(chunk_id)
        
        if success:
            return ChunkOperationResponse(
                status="deleted",
                chunk_id=chunk_id,
                message=f"Deleted chunk {chunk_id}"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to delete chunk")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chunk: {str(e)}")


@router.get("/{chunk_id}/context", response_model=ProcessingContextResponse)
async def get_chunk_context(chunk_id: str):
    """Get processing context for chunk (previous/next chunks)"""
    try:
        service = get_writer_service()
        chunk = service.get_chunk_by_id(chunk_id)
        
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        context = service.get_processing_context(chunk)
        
        if 'error' in context:
            raise HTTPException(status_code=500, detail=context['error'])
        
        return ProcessingContextResponse(
            chunk=_chunk_to_response(context['chunk']),
            previous_chunk=_chunk_to_response(context['previous_chunk']) if context.get('previous_chunk') else None,
            next_chunk=_chunk_to_response(context['next_chunk']) if context.get('next_chunk') else None,
            previous_summaries=context.get('previous_summaries', []),
            next_titles=context.get('next_titles', []),
            position=context.get('position', 'unknown')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chunk context: {str(e)}")


@router.get("/{chunk_id}/neighbors")
async def get_chunk_neighbors(chunk_id: str, max_neighbors: int = Query(5, ge=1, le=20)):
    """Get semantically similar chunks (neighbors)"""
    try:
        service = get_writer_service()
        chunk = service.get_chunk_by_id(chunk_id)
        
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        # Get semantic neighbors using FAISS
        neighbors = service.faiss_manager.get_chunk_neighbors(chunk_id, max_neighbors)
        
        neighbor_chunks = []
        for neighbor_id, similarity in neighbors:
            neighbor_chunk = service.get_chunk_by_id(neighbor_id)
            if neighbor_chunk and neighbor_chunk.id != chunk_id:
                neighbor_chunks.append({
                    "chunk": _chunk_to_response(neighbor_chunk),
                    "similarity": round(similarity, 3)
                })
        
        return {
            "chunk_id": chunk_id,
            "neighbors": neighbor_chunks,
            "total_found": len(neighbor_chunks)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chunk neighbors: {str(e)}")