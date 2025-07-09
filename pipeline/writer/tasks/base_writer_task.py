import luigi
from abc import abstractmethod
from components.structured_task import StructuredTask


class BaseWriterTask(StructuredTask):
    """Base class for all writer pipeline tasks"""
    
    # Luigi parameters for author and title - inherited by all tasks
    author = luigi.Parameter(default="Stanisław Lem")
    title = luigi.Parameter(default="Meta-Nihilizm Pragmatyczny III Stopnia")
    
    @property
    def pipeline_name(self) -> str:
        return "writer"
    
    @property
    @abstractmethod
    def task_name(self) -> str:
        """Task name - must be implemented by each task"""
        pass