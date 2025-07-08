"""
Structure-based document processing pipeline using Surya OCR
"""

from .tasks import StructureDetector, StructureSplitter

__all__ = ['StructureDetector', 'StructureSplitter']
__version__ = "0.1.0"