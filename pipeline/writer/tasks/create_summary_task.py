import luigi
from pathlib import Path

from pipeline.writer.tasks.base_writer_task import BaseWriterTask
from writer.writer_service import WriterService
from writer.models import ChunkStatus
from pipeline.writer.config_loader import ConfigLoader
from llm import LLMClient


class CreateSummaryTask(BaseWriterTask):
    toc_path = luigi.Parameter()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.task_config = self.config.get_task_config('CreateSummaryTask')
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/create_summary"
        self.toc_short_path = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/{self.config.get_output_config()['toc_short_filename']}"
    
    @property 
    def task_name(self) -> str:
        return "create_summary"
    
    def requires(self):
        from .embed_toc_task import EmbedTOCTask
        return EmbedTOCTask(toc_path=self.toc_path)
    
    def output(self):
        return luigi.LocalTarget(f"{self.output_dir}/completed.flag")
    
    def run(self):
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.toc_short_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Persist task start
        self._persist_task_progress("GLOBAL", "CreateSummaryTask", "STARTED")
        
        try:
            # Initialize WriterService
            writer_service = WriterService()
            
            # Get all chunks (should be NOT_STARTED TOC entries)
            chunks = writer_service.get_chunks_by_status(ChunkStatus.NOT_STARTED)
            
            if not chunks:
                print("⚠️ No chunks found for summary")
                with self.output().open('w') as f:
                    f.write("No chunks found")
                self._persist_task_progress("GLOBAL", "CreateSummaryTask", "COMPLETED")
                return
            
            # Prepare full TOC content for summarization
            toc_content = self._prepare_toc_content(chunks)
            
            # Initialize LLM client with configured model
            llm_client = LLMClient(model=self.task_config['model'])
            
            # Create summary prompt from config
            prompt = self._create_summary_prompt(toc_content)
            
            # Generate summary
            print(f"🧠 Generating summary for {len(chunks)} TOC entries...")
            summary = llm_client.chat(prompt)
            
            # Save summary to toc_short.txt
            with open(self.toc_short_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            # Save detailed log
            log_file = f"{self.output_dir}/summary_log.txt"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"Summary generated for {len(chunks)} chunks\n")
                f.write(f"Model used: {self.task_config['model']}\n")
                f.write(f"Output saved to: {self.toc_short_path}\n")
                f.write(f"Summary length: {len(summary)} characters\n")
                f.write(f"TOC content length: {len(toc_content)} characters\n")
                f.write(f"\nTOC Content:\n{toc_content}\n")
                f.write(f"\nGenerated Summary:\n{summary}\n")
            
            # Create completion flag
            with self.output().open('w') as f:
                f.write(f"Summary created for {len(chunks)} TOC entries")
            
            # Persist task completion
            self._persist_task_progress("GLOBAL", "CreateSummaryTask", "COMPLETED")
            
            print(f"✅ Created summary for {len(chunks)} TOC entries in {self.toc_short_path}")
            
        except Exception as e:
            self._persist_task_progress("GLOBAL", "CreateSummaryTask", "FAILED")
            raise
    
    def _prepare_toc_content(self, chunks) -> str:
        """Prepare TOC content for summarization"""
        lines = []
        # Sort by hierarchical_id to maintain order
        sorted_chunks = sorted(chunks, key=lambda x: x.hierarchical_id)
        
        for chunk in sorted_chunks:
            lines.append(f"{chunk.hierarchical_id} {chunk.title}")
        
        return "\n".join(lines)
    
    def _create_summary_prompt(self, toc_content: str) -> str:
        """Create prompt for TOC summarization from config"""
        prompt_config = self.task_config['prompt']
        
        # Build full prompt with system + user messages
        system_message = prompt_config['system']
        user_prompt = prompt_config['user'].format(toc_content=toc_content)
        
        return f"{system_message}\n\n{user_prompt}"
    
    def _persist_task_progress(self, hierarchical_id: str, task_name: str, status: str):
        """Persist task progress to file"""
        progress_file = self.config.get_progress_file()
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | {hierarchical_id} | {task_name} | {status}\n")