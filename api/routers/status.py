"""
Status endpoints - API info and health checks
"""

from fastapi import APIRouter, HTTPException
from api.main import get_writer_service
from api.models import ServiceStatsResponse


router = APIRouter(tags=["status"])


@router.get("/status")
async def get_api_status():
    """Get API status and basic info"""
    try:
        service = get_writer_service()
        stats = service.get_statistics()
        
        return {
            "status": "ready",
            "service": "WriterService",
            "chunks_loaded": stats["total_chunks"],
            "faiss_available": stats["faiss"]["available"],
            "embedding_model": stats["embeddings"]["model"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service error: {str(e)}")


@router.get("/stats", response_model=ServiceStatsResponse)
async def get_comprehensive_stats():
    """Get comprehensive WriterService statistics"""
    try:
        service = get_writer_service()
        stats = service.get_statistics()
        
        return ServiceStatsResponse(
            total_chunks=stats["total_chunks"],
            status_counts=stats["status_counts"],
            storage_size_mb=stats["storage"].get("storage_size_mb", 0.0),
            faiss_vectors=stats["faiss"].get("total_vectors", 0),
            embedding_model=stats["embeddings"]["model"],
            cache_stats=stats["embeddings"].get("cache_stats", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/cleanup")
async def cleanup_orphaned_data():
    """Cleanup orphaned data across all systems"""
    try:
        service = get_writer_service()
        results = service.cleanup_all()
        
        return {
            "status": "completed",
            "orphaned_files_removed": results["orphaned_files"],
            "orphaned_embeddings_removed": results["orphaned_embeddings"],
            "message": "Cleanup completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")