import luigi
from pathlib import Path

from components.structured_task import StructuredTask
from pipeline.writer.parser import TOCParser
from pipeline.writer.database import DatabaseManager
from pipeline.writer.config_loader import ConfigLoader


class ParseTOCTask(StructuredTask):
    toc_path = luigi.Parameter()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.task_config = self.config.get_task_config('ParseTOCTask')
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/parse_toc"
    
    def output(self):
        return luigi.LocalTarget(f"{self.output_dir}/completed.flag")
    
    def run(self):
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Persist task start
        self._persist_task_progress("GLOBAL", "ParseTOCTask", "STARTED")
        
        try:
            # Read TOC file
            with open(self.toc_path, 'r', encoding='utf-8') as f:
                toc_content = f.read()
            
            # Parse TOC
            parser = TOCParser()
            chunks = parser.parse_toc_content(toc_content)
            
            # Validate hierarchy
            errors = parser.validate_hierarchy(chunks)
            if errors:
                error_file = f"{self.output_dir}/errors.txt"
                with open(error_file, 'w') as f:
                    f.write('\n'.join(errors))
                self._persist_task_progress("GLOBAL", "ParseTOCTask", "FAILED")
                raise ValueError(f"TOC hierarchy errors saved to {error_file}")
            
            # Save to database
            db = DatabaseManager(self.config.get_database_path())
            db.save_chunks(chunks)
            
            # Save summary
            summary_file = f"{self.output_dir}/summary.txt"
            with open(summary_file, 'w') as f:
                f.write(f"Parsed {len(chunks)} chunks from {self.toc_path}\n")
                for chunk in chunks:
                    f.write(f"{chunk.hierarchical_id}: {chunk.title}\n")
            
            # Create completion flag
            with self.output().open('w') as f:
                f.write(f"Completed parsing {len(chunks)} chunks")
            
            # Persist task completion
            self._persist_task_progress("GLOBAL", "ParseTOCTask", "COMPLETED")
            
            print(f"✅ Parsed {len(chunks)} chunks and saved to database")
            
        except Exception as e:
            self._persist_task_progress("GLOBAL", "ParseTOCTask", "FAILED")
            raise
    
    def _persist_task_progress(self, hierarchical_id: str, task_name: str, status: str):
        """Persist task progress to file"""
        progress_file = self.config.get_progress_file()
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | {hierarchical_id} | {task_name} | {status}\n")