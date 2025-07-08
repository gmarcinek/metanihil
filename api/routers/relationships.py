"""
Relationship CRUD endpoints
"""

import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from ..deps import get_store
from ..models import RelationshipCreateRequest, RelationshipUpdateRequest

router = APIRouter(prefix="/relationships", tags=["relationships"])


def _format_relationship(source, target, data):
    """Format relationship data - DRY helper"""
    store = get_store()
    source_entity = store.get_entity_by_id(source)
    target_entity = store.get_entity_by_id(target)
    
    return {
        "id": data.get('id', f"{source}-{target}"),
        "source": source_entity.name if source_entity else source,
        "target": target_entity.name if target_entity else target,
        "source_id": source,
        "target_id": target,
        "relation_type": data.get('relation_type', ''),
        "confidence": data.get('confidence', 0.0),
        "evidence": data.get('evidence', ''),
        "created_at": data.get('created_at', ''),
        "discovery_method": data.get('discovery_method', '')
    }


@router.post("")
async def create_relationship(rel_data: RelationshipCreateRequest):
    """Create new relationship between entities"""
    store = get_store()
    
    # Verify both entities exist
    source_entity = store.get_entity_by_id(rel_data.source_id)
    target_entity = store.get_entity_by_id(rel_data.target_id)
    
    if not source_entity:
        raise HTTPException(status_code=404, detail=f"Source entity {rel_data.source_id} not found")
    if not target_entity:
        raise HTTPException(status_code=404, detail=f"Target entity {rel_data.target_id} not found")
    
    try:
        rel_id = str(uuid.uuid4())
        
        # Add relationship to graph
        store.relationship_manager.graph.add_edge(
            rel_data.source_id,
            rel_data.target_id,
            id=rel_id,
            relation_type=rel_data.relation_type,
            confidence=max(0.0, min(1.0, rel_data.confidence or 1.0)),
            evidence=rel_data.evidence or "",
            created_at=datetime.now().isoformat(),
            discovery_method="manual"
        )
        
        print(f"✅ Relationship created: {rel_id} - {source_entity.name} -> {target_entity.name}")
        
        return {
            "status": "created",
            "relationship_id": rel_id,
            "relationship": _format_relationship(
                rel_data.source_id, 
                rel_data.target_id, 
                {
                    "id": rel_id,
                    "relation_type": rel_data.relation_type,
                    "confidence": rel_data.confidence or 1.0,
                    "evidence": rel_data.evidence or "",
                    "created_at": datetime.now().isoformat(),
                    "discovery_method": "manual"
                }
            )
        }
        
    except Exception as e:
        print(f"❌ Failed to create relationship: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create relationship: {e}")


@router.get("/{relationship_id}")
async def get_relationship(relationship_id: str):
    """Get specific relationship details"""
    store = get_store()
    
    # Find relationship in graph
    for source, target, data in store.relationship_manager.graph.edges(data=True):
        if data.get('id') == relationship_id:
            return _format_relationship(source, target, data)
    
    raise HTTPException(status_code=404, detail="Relationship not found")


@router.get("")
async def get_relationships(
    entity_id: Optional[str] = None,
    relation_type: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0)
):
    """Get relationships with optional filtering"""
    store = get_store()
    
    relationships = []
    
    for source, target, data in store.relationship_manager.graph.edges(data=True):
        # Filter by entity_id if provided
        if entity_id and entity_id not in [source, target]:
            continue
            
        # Filter by relation_type if provided
        if relation_type and data.get('relation_type', '').upper() != relation_type.upper():
            continue
        
        relationships.append(_format_relationship(source, target, data))
    
    # Apply pagination
    paginated = relationships[offset:offset+limit]
    
    return {
        "relationships": paginated,
        "total_found": len(relationships),
        "returned": len(paginated)
    }


@router.put("/{relationship_id}")
async def update_relationship(relationship_id: str, update_data: RelationshipUpdateRequest):
    """Update relationship"""
    store = get_store()
    
    # Find and update relationship in graph
    for source, target, data in store.relationship_manager.graph.edges(data=True):
        if data.get('id') == relationship_id:
            try:
                # Update fields if provided
                if update_data.relation_type is not None:
                    data['relation_type'] = update_data.relation_type
                if update_data.confidence is not None:
                    data['confidence'] = max(0.0, min(1.0, update_data.confidence))
                if update_data.evidence is not None:
                    data['evidence'] = update_data.evidence
                
                # Update timestamp
                data['updated_at'] = datetime.now().isoformat()
                
                print(f"✅ Relationship {relationship_id} updated")
                
                return {
                    "status": "updated",
                    "relationship_id": relationship_id,
                    "updated_fields": {
                        k: v for k, v in update_data.dict().items() 
                        if v is not None
                    }
                }
                
            except Exception as e:
                print(f"❌ Failed to update relationship {relationship_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to update relationship: {e}")
    
    raise HTTPException(status_code=404, detail="Relationship not found")


@router.delete("/{relationship_id}")
async def delete_relationship(relationship_id: str):
    """Delete relationship"""
    store = get_store()
    
    # Find and remove relationship from graph
    for source, target, data in store.relationship_manager.graph.edges(data=True):
        if data.get('id') == relationship_id:
            try:
                store.relationship_manager.graph.remove_edge(source, target)
                
                print(f"✅ Relationship {relationship_id} deleted")
                
                return {
                    "status": "deleted",
                    "relationship_id": relationship_id
                }
                
            except Exception as e:
                print(f"❌ Failed to delete relationship {relationship_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to delete relationship: {e}")
    
    raise HTTPException(status_code=404, detail="Relationship not found")