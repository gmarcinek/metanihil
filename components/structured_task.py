import luigi
import hashlib
from pathlib import Path
from abc import ABC, abstractmethod


class StructuredTask(luigi.Task, ABC):
    """
    Base class for Luigi tasks with standardized output structure
    
    Output: output/{document_name}/{task_name}/{task_name}.json
    """
    
    @property
    @abstractmethod
    def pipeline_name(self) -> str:
        """Pipeline name (e.g. 'preprocessing', 'postprocessing')"""
        pass
    
    @property
    @abstractmethod 
    def task_name(self) -> str:
        """Task name (e.g. 'file_router', 'pdf_processing')"""
        pass
    
    def output(self):
        """Document-centric output path with UTF-8 encoding"""
        # Try to get document name from file_path parameter
        doc_name = self._get_document_name()
        
        if doc_name:
            output_dir = Path("output") / doc_name / self.task_name
        else:
            # Fallback to old structure if no file_path
            output_dir = Path("output") / self.pipeline_name / self.task_name
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.task_name}.json"
        return luigi.LocalTarget(str(output_dir / filename), format=luigi.format.UTF8)
    
    def _get_document_name(self):
        """Extract normalized document name from file_path parameter"""
        if hasattr(self, 'file_path') and self.file_path:
            doc_name = Path(self.file_path).stem
            return self._normalize_doc_name(doc_name)
        return None
    
    def _normalize_doc_name(self, doc_name):
        """Normalize document name for filesystem"""
        import re
        # Replace problematic chars with underscores
        normalized = re.sub(r'[<>:"/\\|?*\s]', '_', doc_name)
        # Remove multiple underscores
        normalized = re.sub(r'_+', '_', normalized)
        # Strip leading/trailing underscores
        return normalized.strip('_')
    
    @abstractmethod
    def run(self):
        """Task implementation"""
        pass