from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseOCRClient(ABC):
    """Base interface for OCR clients"""
    
    @abstractmethod
    def process_pages(self, images: List, languages: List[str] = None) -> List[Dict[str, Any]]:
        """
        Process images and return OCR results
        
        Args:
            images: List of image data (PIL, base64, paths)
            languages: Language codes like ['en', 'pl']
            
        Returns:
            List of dicts with OCR results per image
        """
        pass