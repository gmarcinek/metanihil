from dataclasses import dataclass
from enum import Enum
from typing import Optional


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