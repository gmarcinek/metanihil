"""
Entity CRUD endpoints
"""

from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ner.storage.models import StoredEntity, create_entity_id
from ..deps import get_store
from ..models import EntityResponse, EntityCreateRequest, EntityUpdateRequest

def _to_entity_response(entity: StoredEntity) -> EntityResponse:
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



router = APIRouter(prefix="/entities", tags=["entities"])


@router.get("", response_model=List[EntityResponse])
async def get_entities(
    limit: int = Query(50, le=200), 
    offset: int = Query(0, ge=0), 
    entity_type: Optional[str] = None
):
    """Get entities with pagination and optional type filtering"""
    store = get_store()
    entities = list(store.entities.values())
    
    if entity_type:
        entities = [e for e in entities if e.type.upper() == entity_type.upper()]
    
    paginated = entities[offset:offset+limit]
    return [_to_entity_response(e) for e in paginated]


@router.post("")
async def create_entity(entity_data: EntityCreateRequest):
    """Create new entity"""
    store = get_store()
    
    try:        
        # Generate unique ID
        entity_id = create_entity_id(entity_data.name, entity_data.type)
        
        # Create new entity
        entity = StoredEntity(
            id=entity_id,
            name=entity_data.name,
            type=entity_data.type,
            description=entity_data.description or "",
            aliases=entity_data.aliases or [],
            confidence=max(0.0, min(1.0, entity_data.confidence or 1.0)),
            context=entity_data.context or "",
            source_chunk_ids=[],
            document_sources=[],
            merge_count=0
        )
        
        # Save to disk and memory
        store.persistence.save_entity(entity)
        store.entities[entity_id] = entity
        
        print(f"✅ Entity created: {entity_id} - {entity.name}")
        
        return {
            "status": "created",
            "entity_id": entity_id,
            "entity": _to_entity_response(entity)
        }
        
    except Exception as e:
        print(f"❌ Failed to create entity: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create entity: {e}")


@router.get("/{entity_id}")
async def get_entity(entity_id: str):
    """Get specific entity with relationships and related entities"""
    store = get_store()
    entity = store.get_entity_by_id(entity_id)
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    relationships = store.relationship_manager.get_entity_relationships(entity_id)
    related_ids = store.relationship_manager.get_related_entities(entity_id, max_depth=2)
    related = [
        {
            "id": e.id,
            "name": e.name,
            "type": e.type,
            "confidence": e.confidence
        }
        for eid in related_ids[:10]
        if (e := store.get_entity_by_id(eid))
    ]
    
    return {
        "entity": _to_entity_response(entity),
        "relationships": relationships[:20],
        "related_entities": related,
        "metadata": {
            "created_at": entity.created_at,
            "updated_at": entity.updated_at,
            "merge_count": entity.merge_count
        }
    }


@router.put("/{entity_id}")
async def update_entity(entity_id: str, update_data: EntityUpdateRequest):
    """Update entity fields"""
    store = get_store()
    entity = store.get_entity_by_id(entity_id)
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    try:
        # Update entity fields if provided
        if update_data.name is not None:
            entity.name = update_data.name
        if update_data.description is not None:
            entity.description = update_data.description
        if update_data.aliases is not None:
            entity.aliases = update_data.aliases
        if update_data.type is not None:
            entity.type = update_data.type
        if update_data.confidence is not None:
            entity.confidence = max(0.0, min(1.0, update_data.confidence))
        
        # Update timestamp
        entity.updated_at = datetime.now().isoformat()
        
        # Save to disk
        store.persistence.save_entity(entity)
        
        # Update in-memory store
        store.entities[entity_id] = entity
        
        print(f"✅ Entity {entity_id} updated: {entity.name}")
        
        return {
            "status": "updated", 
            "entity_id": entity_id,
            "updated_fields": {
                k: v for k, v in update_data.dict().items() 
                if v is not None
            }
        }
        
    except Exception as e:
        print(f"❌ Failed to update entity {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update entity: {e}")


@router.delete("/{entity_id}")
async def delete_entity(entity_id: str):
    """Delete entity"""
    store = get_store()
    entity = store.get_entity_by_id(entity_id)
    
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    try:
        store.faiss_manager.remove_entity(entity_id)
        if store.relationship_manager.graph.has_node(entity_id):
            store.relationship_manager.graph.remove_node(entity_id)
        del store.entities[entity_id]
        file = store.persistence.entities_dir / f"{entity_id}.json"
        if file.exists():
            file.unlink()
        return {"status": "deleted", "entity_id": entity_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete entity: {e}")


class EntityTypeCount(BaseModel):
    type: str
    count: int


@router.get("/types", response_model=List[EntityTypeCount])
async def get_entity_types():
    """Get all entity types with counts"""
    store = get_store()
    counter = {}
    for e in store.entities.values():
        counter[e.type] = counter.get(e.type, 0) + 1
    
    return [
        EntityTypeCount(type=k, count=v) 
        for k, v in sorted(counter.items())
    ]