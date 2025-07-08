from typing import List, Dict, Any, Optional
from .base import BaseOCRClient
from .surya_manager import SuryaModelManager


class SuryaClient(BaseOCRClient):
    """Surya OCR client - returns raw output, formatting in consumer"""
    
    def __init__(self, languages: List[str] = None, default_task: str = 'ocr_with_boxes'):
        self.languages = languages or ['en']
        self.default_task = default_task
        self.manager = SuryaModelManager()
        self.models = None  # Lazy load
    
    def _ensure_models(self):
        """Lazy load models once"""
        if self.models is None:
            self.models = self.manager.get_models()
    
    def process_pages(self, images: List, languages: List[str] = None) -> List[Dict[str, Any]]:
        """Process images - return raw Surya output"""
        # Sanity check
        if not images or not all(hasattr(img, 'size') for img in images):
            print("⚠️ Invalid or empty image list")
            return []
        
        try:
            self._ensure_models()
            
            print(f"🔍 Processing {len(images)} images with Surya...")
            
            # Layout detection only - most common use case
            layout_predictions = self.models['layout_predictor'](images)
            
            # Safe formatting with None checks
            return [
                {
                    "layout": layout_predictions[i] if (i < len(layout_predictions) and layout_predictions[i]) else [],
                    "page_index": i
                }
                for i in range(len(images))
            ]
            
        except Exception as e:
            print(f"❌ Surya failed: {e}")
            return [{"error": str(e), "page_index": i} for i in range(len(images))]
    
    def process_pages_with_ocr(self, images: List, languages: List[str] = None, task_name: str = None) -> List[Dict[str, Any]]:
        """If OCR needed - separate method"""
        # Sanity check
        if not images or not all(hasattr(img, 'size') for img in images):
            print("⚠️ Invalid or empty image list")
            return []
        
        try:
            self._ensure_models()
            use_task = task_name or self.default_task
            
            print(f"🔍 Processing {len(images)} images with OCR + layout...")
            
            # OCR + Layout
            task_names = [use_task] * len(images)
            ocr_predictions = self.models['recognition_predictor'](images, task_names, self.models['detection_predictor'])
            layout_predictions = self.models['layout_predictor'](images)
            
            # Raw output with None safety
            return [
                {
                    "ocr": ocr_predictions[i] if (i < len(ocr_predictions) and ocr_predictions[i]) else None,
                    "layout": layout_predictions[i] if (i < len(layout_predictions) and layout_predictions[i]) else [],
                    "page_index": i
                }
                for i in range(len(images))
            ]
            
        except Exception as e:
            print(f"❌ Surya OCR failed: {e}")
            return [{"error": str(e), "page_index": i} for i in range(len(images))]
    
    def process_single_page(self, image) -> Dict[str, Any]:
        """Debug single page"""
        results = self.process_pages([image])
        return results[0] if results else {"error": "No result", "page_index": 0}