"""
Text processing endpoints
"""

from fastapi import APIRouter, HTTPException

from ner import process_text_to_knowledge
from ner.loaders import LoadedDocument
from ..deps import get_store
from ..models import TextInput

router = APIRouter(prefix="/process", tags=["processing"])


@router.post("")
async def process_text(input_data: TextInput):
    """Process text and extract entities using NER"""
    store = get_store()
    
    try:
        before = len(store.entities)
        
        doc = LoadedDocument(
            content=input_data.text,
            source_file="api_input",
            file_type="text",
            metadata={"source": "api"}
        )
        
        result = process_text_to_knowledge(
            doc,
            entities_dir="semantic_store",
            model=input_data.model,
            domain_names=input_data.domains,
            output_aggregated=False
        )
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result.get("error", "Processing failed"))
        
        after = len(store.entities)
        stats = result.get("processing_stats", {}).get("extraction_stats", {})
        
        return {
            "status": "success",
            "text_length": len(input_data.text),
            "entities_before": before,
            "entities_after": after,
            "new_entities": after - before,
            "domains_used": input_data.domains,
            "model_used": input_data.model,
            "chunks_created": result.get("processing_stats", {}).get("chunks_created", 0),
            "processing_time_stats": {
                "chunks_processed": stats.get("chunks_processed", 0),
                "entities_extracted_raw": stats.get("entities_extracted_raw", 0),
                "entities_extracted_valid": stats.get("entities_extracted_valid", 0),
                "semantic_deduplication_hits": stats.get("semantic_deduplication_hits", 0)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))