import luigi
from pathlib import Path

from pipeline.writer.tasks.base_writer_task import BaseWriterTask
from writer.writer_service import WriterService
from writer.models import ChunkStatus, ChunkData
from pipeline.writer.config_loader import ConfigLoader
from llm import LLMClient


class ProcessChunksTask(BaseWriterTask):
    toc_path = luigi.Parameter()
    batch_size = luigi.IntParameter(default=5)
    iteration = luigi.IntParameter(default=1)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.task_config = self.config.get_task_config('ProcessChunksTask')
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/process_chunks/iteration_{self.iteration}"
    
    @property 
    def task_name(self) -> str:
        return "proces_chunks"
    
    def requires(self):
        from .create_summary_task import CreateSummaryTask
        return CreateSummaryTask(toc_path=self.toc_path)
    
    def output(self):
        return luigi.LocalTarget(f"{self.output_dir}/completed.flag")
    
    def run(self):
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Persist task start
        self._persist_task_progress("GLOBAL", f"ProcessChunksTask_Iteration_{self.iteration}", "STARTED")
        
        try:
            # Initialize WriterService
            writer_service = WriterService()
            
            # Initialize LLM client
            llm_client = LLMClient(model=self.task_config['model'])
            
            # Get batch of chunks to process
            chunks_to_process = writer_service.get_chunks_batch(ChunkStatus.NOT_STARTED, self.batch_size)
            
            if not chunks_to_process:
                print(f"✅ No more chunks to process in iteration {self.iteration}")
                with self.output().open('w') as f:
                    f.write("No chunks to process")
                self._persist_task_progress("GLOBAL", f"ProcessChunksTask_Iteration_{self.iteration}", "COMPLETED")
                return
            
            # Load TOC summary for context
            toc_summary = self._load_toc_summary()
            
            # Process each chunk in batch
            processed_count = 0
            for chunk in chunks_to_process:
                self._persist_task_progress(chunk.hierarchical_id, f"ProcessChunksTask_Iteration_{self.iteration}", "IN_PROGRESS")
                
                try:
                    # Get processing context for this chunk
                    context = writer_service.get_processing_context(chunk)
                    
                    if 'error' in context:
                        raise ValueError(context['error'])
                    
                    # Generate content for chunk
                    content = self._generate_chunk_content(llm_client, chunk, context, toc_summary)
                    
                    # Generate summary of the content
                    summary = self._generate_chunk_summary(llm_client, chunk, content)
                    
                    # Update chunk with content and summary
                    success = writer_service.update_chunk_content(
                        chunk_id=chunk.id,
                        content=content,
                        summary=summary
                    )
                    
                    if success:
                        # Save individual chunk output
                        self._save_chunk_output(chunk, content, summary)
                        
                        processed_count += 1
                        self._persist_task_progress(chunk.hierarchical_id, f"ProcessChunksTask_Iteration_{self.iteration}", "COMPLETED")
                        print(f"✅ Processed chunk {chunk.hierarchical_id}: {chunk.title}")
                    else:
                        raise ValueError("Failed to update chunk in WriterService")
                    
                except Exception as e:
                    # Mark chunk as failed
                    chunk.status = ChunkStatus.FAILED
                    writer_service.persistence.save_chunk(chunk)
                    writer_service.chunks[chunk.id] = chunk
                    
                    self._persist_task_progress(chunk.hierarchical_id, f"ProcessChunksTask_Iteration_{self.iteration}", "FAILED")
                    print(f"❌ Failed to process chunk {chunk.hierarchical_id}: {str(e)}")
            
            # Create completion flag
            with self.output().open('w') as f:
                f.write(f"Processed {processed_count}/{len(chunks_to_process)} chunks in iteration {self.iteration}")
            
            # Persist task completion
            self._persist_task_progress("GLOBAL", f"ProcessChunksTask_Iteration_{self.iteration}", "COMPLETED")
            
            print(f"✅ Completed iteration {self.iteration}: {processed_count}/{len(chunks_to_process)} chunks processed")
            
        except Exception as e:
            self._persist_task_progress("GLOBAL", f"ProcessChunksTask_Iteration_{self.iteration}", "FAILED")
            raise
    
    def _load_toc_summary(self) -> str:
        """Load TOC summary from previous task"""
        toc_short_path = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/{self.config.get_output_config()['toc_short_filename']}"
        
        try:
            with open(toc_short_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"⚠️ TOC summary not found at {toc_short_path}")
            return "TOC summary not available"
    
    def _generate_chunk_content(self, llm_client: LLMClient, chunk: ChunkData, context: dict, toc_summary: str) -> str:
        """Generate content for chunk using LLM"""
        prompt_config = self.task_config['content_prompt']
        
        # Format context for prompt
        context_text = self._format_context_for_prompt(context, toc_summary)
        
        # Create prompt
        user_prompt = prompt_config['user'].format(
            hierarchical_id=chunk.hierarchical_id,
            title=chunk.title,
            context=context_text
        )
        
        full_prompt = f"{prompt_config['system']}\n\n{user_prompt}"
        
        return llm_client.chat(full_prompt)
    
    def _generate_chunk_summary(self, llm_client: LLMClient, chunk: ChunkData, content: str) -> str:
        """Generate summary of chunk content"""
        prompt_config = self.task_config['summary_prompt']
        
        user_prompt = prompt_config['user'].format(
            hierarchical_id=chunk.hierarchical_id,
            title=chunk.title,
            content=content
        )
        
        full_prompt = f"{prompt_config['system']}\n\n{user_prompt}"
        
        return llm_client.chat(full_prompt)
    
    def _format_context_for_prompt(self, context: dict, toc_summary: str) -> str:
        """Format context information for LLM prompt"""
        parts = [f"HIERARCHIA: {context['chunk'].hierarchical_id}"]
        
        if toc_summary:
            parts.append(f"STRESZCZENIE CAŁOŚCI:\n{toc_summary}")
        
        if context.get('previous_chunk') and context['previous_chunk'].content:
            parts.append(f"POPRZEDNI FRAGMENT:\n{context['previous_chunk'].content}")
        
        if context.get('previous_summaries'):
            summaries = '\n'.join(context['previous_summaries'])
            parts.append(f"POPRZEDNIE STRESZCZENIA:\n{summaries}")
        
        if context.get('next_titles'):
            titles = '\n'.join(context['next_titles'])
            parts.append(f"NASTĘPNE TEMATY:\n{titles}")
        
        parts.append(f"POZYCJA W DOKUMENCIE: {context.get('position', 'unknown')}")
        
        return '\n\n'.join(parts)
    
    def _save_chunk_output(self, chunk: ChunkData, content: str, summary: str):
        """Save individual chunk output to file"""
        chunk_dir = f"{self.output_dir}/chunks"
        Path(chunk_dir).mkdir(parents=True, exist_ok=True)
        
        chunk_file = f"{chunk_dir}/{chunk.hierarchical_id.replace('.', '_')}.txt"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(f"ID: {chunk.hierarchical_id}\n")
            f.write(f"TITLE: {chunk.title}\n")
            f.write(f"STATUS: {ChunkStatus.COMPLETED.value}\n")
            f.write(f"ITERATION: {self.iteration}\n")
            f.write(f"LEVEL: {chunk.level}\n")
            f.write(f"PARENT: {chunk.parent_hierarchical_id or 'None'}\n")
            f.write(f"\nCONTENT:\n{content}\n")
            f.write(f"\nSUMMARY:\n{summary}\n")
    
    def _persist_task_progress(self, hierarchical_id: str, task_name: str, status: str):
        """Persist task progress to file"""
        progress_file = self.config.get_progress_file()
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | {hierarchical_id} | {task_name} | {status}\n")