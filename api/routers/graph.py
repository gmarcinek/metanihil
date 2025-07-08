"""
Knowledge graph endpoints
"""

from typing import Optional
from fastapi import APIRouter, Query

from ..deps import get_store

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("")
async def get_knowledge_graph(
    max_nodes: int = Query(200, le=500), 
    max_edges: int = Query(400, le=1000), 
    entity_types: Optional[str] = None
):
    """Get knowledge graph data with optional filtering"""
    store = get_store()
    graph = store.relationship_manager.export_graph_data()
    
    # Filter by entity types if provided
    if entity_types:
        allowed = set(t.strip().upper() for t in entity_types.split(","))
        nodes = [n for n in graph['nodes'] if n.get('type') != 'entity' or n.get('data', {}).get('type', '').upper() in allowed]
        node_ids = set(n['id'] for n in nodes)
        edges = [e for e in graph['edges'] if e['source'] in node_ids and e['target'] in node_ids]
    else:
        nodes = graph['nodes']
        edges = graph['edges']
    
    return {
        "nodes": nodes[:max_nodes],
        "edges": edges[:max_edges],
        "stats": {
            **graph['stats'],
            "returned_nodes": len(nodes[:max_nodes]),
            "returned_edges": len(edges[:max_edges])
        },
        "truncated": {
            "nodes": len(nodes) > max_nodes,
            "edges": len(edges) > max_edges
        },
        "available_entity_types": list(set(
            n.get('data', {}).get('type', 'unknown')
            for n in nodes if n.get('type') == 'entity'
        ))
    }