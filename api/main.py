"""
FastAPI main application with WriterService integration
"""

import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.routers import document
from writer.writer_service import WriterService

# Global WriterService instance
writer_service: WriterService = None


def get_writer_service() -> WriterService:
    """Get current WriterService instance"""
    global writer_service
    if writer_service is None:
        raise HTTPException(status_code=500, detail="WriterService not initialized")
    return writer_service


# Create FastAPI app
app = FastAPI(title="Writer API", version="1.0.0")


@app.on_event("startup")
async def startup():
    """Initialize WriterService on startup"""
    global writer_service
    print("🚀 Initializing WriterService...")
    writer_service = WriterService(storage_dir="output/writer_storage")
    print(f"✅ Ready with {len(writer_service.chunks)} chunks")


@app.on_event("shutdown") 
async def shutdown():
    """Close WriterService on shutdown"""
    global writer_service
    if writer_service:
        writer_service.close()
    print("👋 WriterService closed")


# Include routers
from .routers import status, processing, chunks, search
app.include_router(status.router)
app.include_router(processing.router)
app.include_router(chunks.router)
app.include_router(search.router)
app.include_router(document.router)


@app.get("/")
async def root():
    """API info"""
    service = get_writer_service()
    stats = service.get_statistics()
    
    return {
        "message": "Writer API",
        "chunks": stats["total_chunks"],
        "status": "ready"
    }


@app.post("/reload")
async def reload_service():
    """Reload WriterService data - manual refresh"""
    global writer_service
    
    try:
        old_count = len(writer_service.chunks) if writer_service else 0
        
        # Create fresh WriterService instance
        writer_service = WriterService(storage_dir="output/writer_storage")
        new_count = len(writer_service.chunks)
        
        return {
            "status": "success",
            "message": "WriterService reloaded",
            "chunks_before": old_count,
            "chunks_after": new_count,
            "new_chunks": new_count - old_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")