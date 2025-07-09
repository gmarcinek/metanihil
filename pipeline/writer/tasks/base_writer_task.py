import luigi
from abc import abstractmethod
from components.structured_task import StructuredTask


class BaseWriterTask(StructuredTask):
    """Base class for all writer pipeline tasks"""
    
    # Luigi parameters for author and title - inherited by all tasks
    author = luigi.Parameter(default="pisarz współczesny")
    title = luigi.Parameter(default="interesujące dzieło")
    custom_prompt = luigi.Parameter(default="")
    
    @property
    def pipeline_name(self) -> str:
        return "writer"
    
    @property
    @abstractmethod
    def task_name(self) -> str:
        """Task name - must be implemented by each task"""
        pass
    
    def format_prompt_template(self, template: str, **extra_vars) -> str:
        """Format prompt template with task parameters and extra variables"""
        format_vars = {
            'author': self.author,
            'title': self.title,
            'custom_prompt': self.custom_prompt
        }
        format_vars.update(extra_vars)
        
        return template.format(**format_vars)