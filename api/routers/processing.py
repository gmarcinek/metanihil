"""
Processing endpoints - TOC parsing, embeddings, and content generation
"""

from fastapi import APIRouter, HTTPException
from api.main import get_writer_service
from api.models import (
    TOCProcessRequest, TOCProcessResponse,
    EmbeddingRequest, EmbeddingResponse,
    ProcessingStatsResponse, SuccessResponse
)
from writer.models import ChunkStatus


router = APIRouter(prefix="/processing", tags=["processing"])


@router.post("/toc", response_model=TOCProcessResponse)
async def process_toc_content(request: TOCProcessRequest):
    """Parse TOC content and create chunks with optional embedding"""
    try:
        service = get_writer_service()
        
        # Parse TOC content
        chunks = service.parse_toc_content(request.toc_content)
        
        if not chunks:
            return TOCProcessResponse(
                status="warning",
                chunks_created=0,
                chunks_embedded=0,
                message="No valid TOC entries found"
            )
        
        # Save chunks
        saved_count = service.save_toc_chunks(chunks)
        
        # Generate embeddings if requested
        embedded_count = 0
        if request.auto_embed:
            chunk_ids = [chunk.id for chunk in chunks]
            embedded_count = service.embed_toc_chunks(chunk_ids)
        
        return TOCProcessResponse(
            status="success",
            chunks_created=saved_count,
            chunks_embedded=embedded_count,
            message=f"Created {saved_count} chunks" + (f", embedded {embedded_count}" if request.auto_embed else "")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TOC processing failed: {str(e)}")


@router.post("/embeddings", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for chunks"""
    try:
        service = get_writer_service()
        
        if request.chunk_ids:
            # Specific chunks
            chunks_to_embed = [service.get_chunk_by_id(cid) for cid in request.chunk_ids if service.get_chunk_by_id(cid)]
            
            if not chunks_to_embed:
                return EmbeddingResponse(
                    status="warning",
                    chunks_processed=0,
                    chunks_embedded=0,
                    message="No valid chunks found for provided IDs"
                )
            
            # Generate embeddings manually
            embedded_count = 0
            for chunk in chunks_to_embed:
                embedding = service.embedder.embed_chunk(chunk, use_content=request.use_content)
                if embedding is not None:
                    chunk.embedding = embedding
                    chunk.update_timestamp()
                    service.faiss_manager.add_chunk_embedding(chunk, embedding)
                    service.persistence.save_chunk(chunk)
                    service.chunks[chunk.id] = chunk
                    embedded_count += 1
            
            # Save FAISS index
            service.faiss_manager.save_index()
            
            return EmbeddingResponse(
                status="success",
                chunks_processed=len(chunks_to_embed),
                chunks_embedded=embedded_count,
                message=f"Generated embeddings for {embedded_count}/{len(chunks_to_embed)} chunks"
            )
        else:
            # All chunks that need embeddings
            embedded_count = service.embed_toc_chunks(force=request.force_regenerate)
            
            return EmbeddingResponse(
                status="success",
                chunks_processed=embedded_count,
                chunks_embedded=embedded_count,
                message=f"Generated {embedded_count} embeddings"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")


@router.get("/stats", response_model=ProcessingStatsResponse)
async def get_processing_stats():
    """Get processing statistics"""
    try:
        service = get_writer_service()
        stats = service.get_statistics()
        
        # Calculate processing stats
        status_counts = stats["status_counts"]
        total = sum(status_counts.values())
        
        return ProcessingStatsResponse(
            total_chunks=total,
            processed_chunks=status_counts.get("completed", 0),
            failed_chunks=status_counts.get("failed", 0),
            skipped_chunks=status_counts.get("in_progress", 0) + status_counts.get("not_started", 0),
            success_rate=round((status_counts.get("completed", 0) / max(total, 1)) * 100, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get processing stats: {str(e)}")


@router.post("/content/{chunk_id}")
async def update_chunk_content(chunk_id: str, content: str, summary: str = None):
    """Update chunk content and regenerate embeddings"""
    try:
        service = get_writer_service()
        
        # Check if chunk exists
        chunk = service.get_chunk_by_id(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        # Update content using WriterService (handles embeddings automatically)
        success = service.update_chunk_content(chunk_id, content, summary)
        
        if success:
            return SuccessResponse(
                status="success",
                message=f"Updated content for chunk {chunk_id}",
                data={
                    "chunk_id": chunk_id,
                    "content_length": len(content),
                    "has_summary": summary is not None,
                    "embedding_updated": True
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to update chunk content")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Content update failed: {str(e)}")


@router.delete("/chunks/{chunk_id}")
async def delete_chunk(chunk_id: str):
    """Delete chunk from all systems"""
    try:
        service = get_writer_service()
        
        # Check if chunk exists
        chunk = service.get_chunk_by_id(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        
        # Delete chunk
        success = service.delete_chunk(chunk_id)
        
        if success:
            return SuccessResponse(
                status="success",
                message=f"Deleted chunk {chunk_id}",
                data={"chunk_id": chunk_id}
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to delete chunk")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunk deletion failed: {str(e)}")


@router.post("/reset")
async def reset_all_data():
    """Reset all data - WARNING: destructive operation"""
    try:
        service = get_writer_service()
        
        # Get current stats
        stats = service.get_statistics()
        chunks_before = stats["total_chunks"]
        
        # Delete all chunks
        chunk_ids = list(service.chunks.keys())
        deleted_count = 0
        
        for chunk_id in chunk_ids:
            if service.delete_chunk(chunk_id):
                deleted_count += 1
        
        # Cleanup orphaned data
        cleanup_results = service.cleanup_all()
        
        return SuccessResponse(
            status="success",
            message=f"Reset completed: deleted {deleted_count} chunks",
            data={
                "chunks_deleted": deleted_count,
                "chunks_before": chunks_before,
                "orphaned_files_cleaned": cleanup_results["orphaned_files"],
                "orphaned_embeddings_cleaned": cleanup_results["orphaned_embeddings"]
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")