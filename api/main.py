"""
FastAPI main application with routers
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from .deps import initialize_store
from .routers import status, entities, relationships, search, processing, graph, analyze

app = FastAPI(title="NER Knowledge API", version="1.0.0")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(status.router)
app.include_router(entities.router)
app.include_router(relationships.router)
app.include_router(search.router)
app.include_router(processing.router)
app.include_router(graph.router)
app.include_router(analyze.router)


@app.on_event("startup")
async def startup():
    """Initialize store and watcher on startup"""
    storage_dir = Path(__file__).parent.parent / "semantic_store"
    initialize_store(storage_dir)
    print("🚀 NER Knowledge API started")