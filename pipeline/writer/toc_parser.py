import re
import uuid
import luigi
import sqlite3
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


class ChunkStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ChunkData:
    id: str
    hierarchical_id: str
    parent_hierarchical_id: Optional[str]
    title: str
    level: int
    display_level: int
    status: ChunkStatus
    content: Optional[str] = None
    summary: Optional[str] = None


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
            
            chunks = []
            for row in rows:
                chunk = ChunkData(
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
                chunks.append(chunk)
            
            return chunks


class TOCParser:
    def __init__(self):
        self.toc_pattern = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')
    
    def _generate_short_uuid(self) -> str:
        """Generate short 8-character UUID"""
        return uuid.uuid4().hex[:8]
    
    def _extract_parent_id(self, hierarchical_id: str) -> Optional[str]:
        """Extract parent hierarchical ID by removing last segment"""
        segments = hierarchical_id.split('.')
        if len(segments) <= 1:
            return None
        return '.'.join(segments[:-1])
    
    def _calculate_level(self, hierarchical_id: str) -> int:
        """Calculate actual level depth"""
        return len(hierarchical_id.split('.'))
    
    def _calculate_display_level(self, level: int) -> int:
        """Calculate display level (max 3)"""
        return min(level, 3)
    
    def parse_toc_content(self, content: str) -> List[ChunkData]:
        """Parse TOC content string and return list of ChunkData objects"""
        chunks = []
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            match = self.toc_pattern.match(line)
            if not match:
                continue
                
            hierarchical_id = match.group(1)
            title = match.group(2).strip()
            
            level = self._calculate_level(hierarchical_id)
            display_level = self._calculate_display_level(level)
            parent_id = self._extract_parent_id(hierarchical_id)
            
            chunk = ChunkData(
                id=self._generate_short_uuid(),
                hierarchical_id=hierarchical_id,
                parent_hierarchical_id=parent_id,
                title=title,
                level=level,
                display_level=display_level,
                status=ChunkStatus.NOT_STARTED
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def validate_hierarchy(self, chunks: List[ChunkData]) -> List[str]:
        """Validate that all parent IDs exist in the chunk list"""
        hierarchical_ids = {chunk.hierarchical_id for chunk in chunks}
        errors = []
        
        for chunk in chunks:
            if chunk.parent_hierarchical_id and chunk.parent_hierarchical_id not in hierarchical_ids:
                errors.append(f"Missing parent '{chunk.parent_hierarchical_id}' for chunk '{chunk.hierarchical_id}'")
        
        return errors


class ParseTOCTask(luigi.Task):
    toc_path = luigi.Parameter()
    
    def output(self):
        return luigi.LocalTarget("data/toc_parsed.flag")
    
    def run(self):
        # Read TOC file
        with open(self.toc_path, 'r', encoding='utf-8') as f:
            toc_content = f.read()
        
        # Parse TOC
        parser = TOCParser()
        chunks = parser.parse_toc_content(toc_content)
        
        # Validate hierarchy
        errors = parser.validate_hierarchy(chunks)
        if errors:
            raise ValueError(f"TOC hierarchy errors: {errors}")
        
        # Save to database
        db = DatabaseManager()
        db.save_chunks(chunks)
        
        # Create completion flag
        with self.output().open('w') as f:
            f.write(f"Parsed {len(chunks)} chunks from {self.toc_path}")
        
        print(f"✅ Parsed {len(chunks)} chunks and saved to database")