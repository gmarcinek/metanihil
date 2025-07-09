from typing import List, Tuple, Dict, Optional, Any
from .models import ChunkData


def parse_hierarchical_id(hierarchical_id: str) -> Tuple[int, ...]:
    """Parse hierarchical ID for numerical sorting (1.10.2 after 1.2.1)"""
    parts = hierarchical_id.split('.')
    return tuple(int(part) for part in parts)


def sort_chunks_numerically(chunks: List[ChunkData]) -> List[ChunkData]:
    """Sort chunks by numerical hierarchical ID"""
    return sorted(chunks, key=lambda x: parse_hierarchical_id(x.hierarchical_id))


def get_level1_group(hierarchical_id: str) -> str:
    """Extract level 1 group from hierarchical ID (e.g., '1.2.3' -> '1')"""
    return hierarchical_id.split('.')[0]


def chunks_in_group(chunks: List[ChunkData], group_id: str) -> List[ChunkData]:
    """Filter chunks belonging to specific level-1 group"""
    return [c for c in chunks if get_level1_group(c.hierarchical_id) == group_id]


def find_chunk_index(chunks: List[ChunkData], target_chunk: ChunkData) -> Optional[int]:
    """Find index of target chunk in sorted list"""
    for i, chunk in enumerate(chunks):
        if chunk.hierarchical_id == target_chunk.hierarchical_id:
            return i
    return None


def get_main_group_chunk(chunks: List[ChunkData], group_id: str) -> Optional[ChunkData]:
    """Get main chunk for group (e.g., chunk with ID '1' for group '1')"""
    return next((c for c in chunks if c.hierarchical_id == group_id), None)


def format_chunk_reference(chunk: ChunkData, include_summary: bool = False) -> str:
    """Format chunk reference for context"""
    base = f"{chunk.hierarchical_id} {chunk.title}"
    if include_summary and chunk.summary:
        return f"{base}: {chunk.summary}"
    return base


def get_previous_groups(current_group_id: str) -> List[str]:
    """Get all previous level-1 group IDs"""
    current_num = int(current_group_id)
    return [str(i) for i in range(current_num)]


def slice_chunks_safely(chunks: List[ChunkData], start: int, end: int) -> List[ChunkData]:
    """Safely slice chunks list with bounds checking"""
    start = max(0, start)
    end = min(len(chunks), end)
    return chunks[start:end] if start < end else []