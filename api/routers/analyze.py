"""
Context analysis endpoints - find similar entities and relationships for text input
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..deps import get_store

router = APIRouter(prefix="/analyze", tags=["analyze"])


class ContextAnalysisRequest(BaseModel):
    """Request model for context analysis"""
    text: str
    similarity_threshold: Optional[float] = 0.6
    max_entities: Optional[int] = 10
    include_relationships: Optional[bool] = True


class EntityWithRelationships(BaseModel):
    """Entity with its relationships"""
    entity_info: str  # "Name - description" format
    relationships: List[str]  # ["Name TYPE_RELATION Name"] format
    similarity_score: float


class ContextAnalysisResponse(BaseModel):
    """Response model for human-readable context analysis"""
    query_text: str
    matched_entities: List[EntityWithRelationships]
    total_entities_found: int
    query_stats: dict


@router.post("/context", response_model=ContextAnalysisResponse)
async def analyze_context(request: ContextAnalysisRequest):
    """
    Analyze text context and return similar entities with relationships in human-readable format
    """
    store = get_store()
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Get contextual entities using semantic similarity
        contextual_entities = store.get_contextual_entities_for_ner(
            request.text, 
            max_entities=request.max_entities,
            threshold=request.similarity_threshold  # ← PASS USER THRESHOLD
        )
        
        # Get full entity objects and calculate similarities
        matched_entities = []
        
        for entity_data in contextual_entities:
            # Find full entity in store by name (since get_contextual_entities_for_ner returns limited data)
            full_entity = None
            for entity_id, entity in store.entities.items():
                if entity.name == entity_data['name']:
                    full_entity = entity
                    break
            
            if not full_entity:
                continue
            
            # Get entity embedding and calculate similarity score
            if full_entity.context_embedding is not None:
                query_embedding = store.embedder._get_cached_embedding(request.text, "temp_query")
                similarity = store.embedder.compute_similarity(
                    query_embedding, 
                    full_entity.context_embedding
                )
                
                # Filter by threshold
                if similarity < request.similarity_threshold:
                    continue
                
                # Format entity info
                entity_info = f"{full_entity.name} - {full_entity.description}"
                
                # Get relationships if requested
                relationships = []
                if request.include_relationships:
                    entity_relationships = store.relationship_manager.get_entity_relationships(full_entity.id)
                    
                    for rel in entity_relationships:
                        source_entity = store.get_entity_by_id(rel['source'])
                        target_entity = store.get_entity_by_id(rel['target'])
                        
                        if source_entity and target_entity:
                            rel_str = f"{source_entity.name} {rel.get('relation_type', 'RELATED')} {target_entity.name}"
                            relationships.append(rel_str)
                
                matched_entities.append(EntityWithRelationships(
                    entity_info=entity_info,
                    relationships=relationships,
                    similarity_score=round(similarity, 3)
                ))
        
        # Sort by similarity score (highest first)
        matched_entities.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return ContextAnalysisResponse(
            query_text=request.text,
            matched_entities=matched_entities,
            total_entities_found=len(matched_entities),
            query_stats={
                "similarity_threshold": request.similarity_threshold,
                "max_entities_requested": request.max_entities,
                "total_entities_in_store": len(store.entities)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Context analysis failed: {str(e)}")


@router.post("/context/prompt")
async def analyze_context_prompt_ready(request: ContextAnalysisRequest):
    """
    Analyze text context and return LLM-ready prompt format
    """
    store = get_store()
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Get contextual entities
        contextual_entities = store.get_contextual_entities_for_ner(
            request.text, 
            max_entities=request.max_entities,
            threshold=request.similarity_threshold  # ← PASS USER THRESHOLD
        )
        
        # Build prompt-ready content
        prompt_lines = []
        prompt_lines.append("KNOWN ENTITIES IN CONTEXT:")
        
        entity_relationships = {}
        entities_processed = 0
        
        for entity_data in contextual_entities:
            # Find full entity in store
            full_entity = None
            for entity_id, entity in store.entities.items():
                if entity.name == entity_data['name']:
                    full_entity = entity
                    break
            
            if not full_entity:
                continue
                
            # Calculate similarity score
            if full_entity.context_embedding is not None:
                query_embedding = store.embedder._get_cached_embedding(request.text, "temp_query")
                similarity = store.embedder.compute_similarity(
                    query_embedding, 
                    full_entity.context_embedding
                )
                
                # Filter by threshold
                if similarity < request.similarity_threshold:
                    continue
                
                # Add entity info
                prompt_lines.append(f"- {full_entity.name}: {full_entity.description}")
                entities_processed += 1
                
                # Collect relationships
                if request.include_relationships:
                    entity_relationships[full_entity.id] = store.relationship_manager.get_entity_relationships(full_entity.id)
        
        # Add relationships section
        if request.include_relationships and entity_relationships:
            prompt_lines.append("")
            prompt_lines.append("ENTITY RELATIONSHIPS:")
            
            for entity_id, relationships in entity_relationships.items():
                for rel in relationships:
                    source_entity = store.get_entity_by_id(rel['source'])
                    target_entity = store.get_entity_by_id(rel['target'])
                    
                    if source_entity and target_entity:
                        rel_str = f"- {source_entity.name} {rel.get('relation_type', 'RELATED')} {target_entity.name}"
                        prompt_lines.append(rel_str)
        
        # Add instruction
        prompt_lines.append("")
        prompt_lines.append("Consider these entities when processing new text for entity extraction.")
        
        # Return as plain text
        prompt_content = "\n".join(prompt_lines)
        
        return {
            "prompt_content": prompt_content,
            "entities_included": entities_processed,
            "relationships_included": sum(len(rels) for rels in entity_relationships.values()),
            "query_stats": {
                "similarity_threshold": request.similarity_threshold,
                "max_entities_requested": request.max_entities,
                "total_entities_in_store": len(store.entities)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt generation failed: {str(e)}")


@router.get("/stats")
async def get_analysis_stats():
    """Get analysis capabilities and store statistics"""
    store = get_store()
    
    # Get entity type distribution
    type_counts = {}
    for entity in store.entities.values():
        entity_type = entity.type
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
    
    # Get relationship type distribution
    rel_counts = {}
    for source, target, data in store.relationship_manager.graph.edges(data=True):
        rel_type = data.get('relation_type', 'unknown')
        rel_counts[rel_type] = rel_counts.get(rel_type, 0) + 1
    
    return {
        "store_stats": {
            "total_entities": len(store.entities),
            "total_chunks": len(store.chunks),
            "total_relationships": store.relationship_manager.graph.number_of_edges()
        },
        "entity_types": type_counts,
        "relationship_types": rel_counts,
        "embedding_model": store.embedder.model_name,
        "analysis_capabilities": {
            "context_analysis": True,
            "similarity_search": True,
            "relationship_discovery": True,
            "prompt_generation": True
        }
    }