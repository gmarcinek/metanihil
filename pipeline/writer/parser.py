import re
import uuid
from typing import List, Optional

from .models import ChunkData, ChunkStatus


class TOCParser:
    def __init__(self):
        self.toc_pattern = re.compile(r'^(\d+(?:\.\d+)*)\s+(.+)$')
    
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
            
            chunk = ChunkData(
                id=self._generate_short_uuid(),
                hierarchical_id=hierarchical_id,
                parent_hierarchical_id=self._extract_parent_id(hierarchical_id),
                title=title,
                level=self._calculate_level(hierarchical_id),
                display_level=self._calculate_display_level(hierarchical_id),
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
    
    def _calculate_display_level(self, hierarchical_id: str) -> int:
        """Calculate display level (max 3)"""
        level = self._calculate_level(hierarchical_id)
        return min(level, 3)