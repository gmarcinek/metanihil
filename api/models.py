"""
Shared Pydantic models for API endpoints
Basic models used across multiple routers
"""

from pydantic import BaseModel
from typing import List, Optional


class EntityResponse(BaseModel):
    """Base entity response model used across multiple endpoints"""
    id: str
    name: str
    type: str
    confidence: float
    aliases: List[str]
    description: str
    source_chunks: List[str]


class TextInput(BaseModel):
    """Text processing input model"""
    text: str
    domains: Optional[List[str]] = ["auto"]
    model: Optional[str] = "gpt-4o-mini"


class SearchQuery(BaseModel):
    """Search query model"""
    query: str
    max_results: Optional[int] = 10


class EntityUpdateRequest(BaseModel):
    """Entity update request model"""
    name: Optional[str] = None
    description: Optional[str] = None
    aliases: Optional[List[str]] = None
    type: Optional[str] = None
    confidence: Optional[float] = None


class EntityCreateRequest(BaseModel):
    """Entity creation request model"""
    name: str
    type: str
    description: Optional[str] = ""
    aliases: Optional[List[str]] = []
    confidence: Optional[float] = 1.0
    context: Optional[str] = ""


class RelationshipCreateRequest(BaseModel):
    """Relationship creation request model"""
    source_id: str
    target_id: str
    relation_type: str
    confidence: Optional[float] = 1.0
    evidence: Optional[str] = ""


class RelationshipUpdateRequest(BaseModel):
    """Relationship update request model"""
    relation_type: Optional[str] = None
    confidence: Optional[float] = None
    evidence: Optional[str] = None