"""
API Dependencies - Store management and file watcher
"""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ner.storage import SemanticStore


# Global state
store: Optional[SemanticStore] = None
store_loaded_at: datetime = datetime.now()
_watcher_lock = threading.Lock()
_last_reload_time = 0


class SemanticStoreChangeHandler(FileSystemEventHandler):
    """File watcher with debouncing to prevent circular reloads"""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.debounce_delay = 2.0  # 2 seconds
    
    def on_any_event(self, event):
        global store, store_loaded_at, _last_reload_time
        
        if event.event_type not in ('created', 'modified', 'deleted', 'moved'):
            return
        
        # Skip temporary files and API's own writes
        if any(skip in str(event.src_path) for skip in ['.tmp', '~', '.swp', '__pycache__']):
            return
        
        current_time = time.time()
        
        with _watcher_lock:
            # Debouncing - ignore rapid successive changes
            if current_time - _last_reload_time < self.debounce_delay:
                return
            
            _last_reload_time = current_time
        
        try:
            print(f"\n🔄 Semantic store change detected: {event.src_path}")
            print("⏳ Reloading store...")
            
            # Reload store
            store = SemanticStore(storage_dir=str(self.storage_dir))
            store_loaded_at = datetime.now()
            
            print(f"✅ Store reloaded: {len(store.entities)} entities, {len(store.chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Failed to reload store: {e}")


def start_watcher(storage_dir: Path) -> None:
    """Start file system watcher with debouncing"""
    def watcher_thread():
        observer = Observer()
        handler = SemanticStoreChangeHandler(storage_dir)
        observer.schedule(handler, str(storage_dir), recursive=True)
        observer.daemon = True
        observer.start()
        print(f"👁️ File watcher started: {storage_dir}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
    
    thread = threading.Thread(target=watcher_thread, daemon=True)
    thread.start()


def get_store() -> SemanticStore:
    """Get current semantic store instance"""
    if store is None:
        raise RuntimeError("Store not initialized")
    return store


def get_store_status() -> dict:
    """Get store status and stats"""
    if store is None:
        return {"status": "not_initialized"}
    
    return {
        "last_loaded": store_loaded_at.isoformat(),
        "entities": len(store.entities),
        "chunks": len(store.chunks),
        "status": "ready"
    }


def initialize_store(storage_dir: Path) -> SemanticStore:
    """Initialize semantic store and start watcher"""
    global store, store_loaded_at
    
    store = SemanticStore(storage_dir=str(storage_dir))
    store_loaded_at = datetime.now()
    
    # Start file watcher
    start_watcher(storage_dir)
    
    print(f"📊 Store initialized: {len(store.entities)} entities, {len(store.chunks)} chunks")
    return store