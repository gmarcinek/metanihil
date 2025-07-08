"""
Search endpoints
"""

from fastapi import APIRouter

from ..deps import get_store
from ..models import SearchQuery, EntityResponse

router = APIRouter(prefix="/search", tags=["search"])


def _to_entity_response(entity) -> EntityResponse:
    """Convert StoredEntity to EntityResponse - DRY helper"""
    return EntityResponse(
        id=entity.id,
        name=entity.name,
        type=entity.type,
        confidence=entity.confidence,
        aliases=entity.aliases,
        description=entity.description,
        source_chunks=entity.source_chunk_ids
    )


@router.post("")
async def search_entities(query: SearchQuery):
    """Search entities by name using semantic similarity"""
    store = get_store()
    results = store.search_entities_by_name(query.query, query.max_results)
    
    return {
        "query": query.query,
        "results": [
            {
                "entity": _to_entity_response(e),
                "similarity": round(sim, 3)
            } for e, sim in results
        ],
        "total_found": len(results)
    }