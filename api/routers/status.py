"""
Status endpoints - basic API info and health checks
"""

from fastapi import APIRouter, HTTPException

from ..deps import get_store, get_store_status

router = APIRouter()


@router.get("/")
async def root():
    """API info and available endpoints"""
    return {
        "message": "NER Knowledge API",
        "version": "1.0.0",
        "endpoints": ["/stats", "/entities", "/search", "/process", "/graph", "/relationships"]
    }


@router.get("/store-status")
async def store_status():
    """Get semantic store status and basic info"""
    return get_store_status()


@router.get("/stats")
async def get_stats():
    """Get comprehensive semantic store statistics"""
    try:
        store = get_store()
        stats = store.get_stats()
        
        # Handle relationships - can be int or dict
        relationships_count = stats.get('relationships', 0)
        if isinstance(relationships_count, dict):
            relationships_count = relationships_count.get('total_relationships', 0)
        elif not isinstance(relationships_count, int):
            relationships_count = 0
        
        return {
            "entities": stats.get('entities', 0),
            "chunks": stats.get('chunks', 0),
            "relationships": relationships_count,
            "storage_size_mb": round(stats.get('storage', {}).get('total_size_mb', 0), 2),
            "embedding_model": stats.get('embedder', {}).get('model_name', 'unknown'),
        }
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Store not initialized")