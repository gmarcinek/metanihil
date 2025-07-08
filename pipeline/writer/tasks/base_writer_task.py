from abc import abstractmethod
from components.structured_task import StructuredTask


class BaseWriterTask(StructuredTask):
    """Base class for all writer pipeline tasks"""
    
    @property
    def pipeline_name(self) -> str:
        return "writer"
    
    @property
    @abstractmethod
    def task_name(self) -> str:
        """Task name - must be implemented by each task"""
        pass