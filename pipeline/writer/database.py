import sqlite3
from typing import List
from pathlib import Path

from .models import ChunkData, ChunkStatus


class DatabaseManager:
    def __init__(self, db_path: str = "data/writer.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id VARCHAR(8) PRIMARY KEY,
                    hierarchical_id VARCHAR UNIQUE NOT NULL,
                    parent_hierarchical_id VARCHAR,
                    title TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    display_level INTEGER NOT NULL,
                    status VARCHAR NOT NULL,
                    content TEXT,
                    summary TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def save_chunks(self, chunks: List[ChunkData]):
        """Save chunks to database"""
        with sqlite3.connect(self.db_path) as conn:
            for chunk in chunks:
                conn.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (id, hierarchical_id, parent_hierarchical_id, title, level, display_level, status, content, summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.id,
                    chunk.hierarchical_id,
                    chunk.parent_hierarchical_id,
                    chunk.title,
                    chunk.level,
                    chunk.display_level,
                    chunk.status.value,
                    chunk.content,
                    chunk.summary
                ))
            conn.commit()
    
    def get_chunks_by_status(self, status: ChunkStatus) -> List[ChunkData]:
        """Get all chunks with specific status"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM chunks WHERE status = ? ORDER BY hierarchical_id",
                (status.value,)
            )
            rows = cursor.fetchall()
            
            return [self._row_to_chunk(row) for row in rows]
    
    def get_chunks_batch(self, status: ChunkStatus, limit: int) -> List[ChunkData]:
        """Get limited number of chunks with specific status"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM chunks WHERE status = ? ORDER BY hierarchical_id LIMIT ?",
                (status.value, limit)
            )
            rows = cursor.fetchall()
            
            return [self._row_to_chunk(row) for row in rows]
    
    def update_chunk_status(self, chunk_id: str, status: ChunkStatus):
        """Update single chunk status"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE chunks SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status.value, chunk_id)
            )
            conn.commit()
    
    def get_chunk_by_hierarchical_id(self, hierarchical_id: str) -> ChunkData:
        """Get chunk by hierarchical ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM chunks WHERE hierarchical_id = ?",
                (hierarchical_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return self._row_to_chunk(row)
            return None
    
    def _row_to_chunk(self, row: sqlite3.Row) -> ChunkData:
        """Convert database row to ChunkData"""
        return ChunkData(
            id=row['id'],
            hierarchical_id=row['hierarchical_id'],
            parent_hierarchical_id=row['parent_hierarchical_id'],
            title=row['title'],
            level=row['level'],
            display_level=row['display_level'],
            status=ChunkStatus(row['status']),
            content=row['content'],
            summary=row['summary']
        )