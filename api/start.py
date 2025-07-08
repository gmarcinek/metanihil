"""
API Server Launcher
"""

import uvicorn
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

def main():
    """Main entry point for poetry script"""
    print("\U0001F680 Starting NER Knowledge API")
    print("\U0001F4CA Server: http://localhost:8000")
    print("\U0001F4D6 Docs: http://localhost:8000/docs") 
    print("\U0001F50D Interactive: http://localhost:8000/redoc")
    print()

    uvicorn.run(
        "api.main:app",  # Changed from api.server:app to api.main:app
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[
            str(Path(__file__).parent),
            str(Path(__file__).parent.parent / "semantic_store" / "entities"),
            str(Path(__file__).parent.parent / "semantic_store" / "chunks"),
        ],
        log_level="info"
    )

if __name__ == "__main__":
    main()