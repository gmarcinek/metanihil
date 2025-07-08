import luigi
from pathlib import Path

from components.structured_task import StructuredTask
from pipeline.writer.database import DatabaseManager
from pipeline.writer.models import ChunkStatus, ChunkData
from pipeline.writer.config_loader import ConfigLoader
from llm import LLMClient


class ProcessChunksTask(StructuredTask):
    toc_path = luigi.Parameter()
    batch_size = luigi.IntParameter(default=5)
    iteration = luigi.IntParameter(default=1)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.task_config = self.config.get_task_config('ProcessChunksTask')
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/process_chunks/iteration_{self.iteration}"
    
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
            # Initialize components
            db = DatabaseManager(self.config.get_database_path())
            llm_client = LLMClient(model=self.task_config['model'])
            
            # Get batch of chunks to process
            chunks_to_process = db.get_chunks_batch(ChunkStatus.NOT_STARTED, self.batch_size)
            
            if not chunks_to_process:
                print(f"✅ No more chunks to process in iteration {self.iteration}")
                with self.output().open('w') as f:
                    f.write("No chunks to process")
                self._persist_task_progress("GLOBAL", f"ProcessChunksTask_Iteration_{self.iteration}", "COMPLETED")
                return
            
            # Get all chunks for context building
            all_chunks = db.get_chunks_by_status(ChunkStatus.NOT_STARTED) + db.get_chunks_by_status(ChunkStatus.COMPLETED)
            all_chunks.sort(key=lambda x: x.hierarchical_id)
            
            # Load TOC summary
            toc_summary = self._load_toc_summary()
            
            # Process each chunk in batch
            processed_count = 0
            for chunk in chunks_to_process:
                self._persist_task_progress(chunk.hierarchical_id, f"ProcessChunksTask_Iteration_{self.iteration}", "IN_PROGRESS")
                
                try:
                    # Find chunk index in all_chunks for context
                    chunk_index = next((i for i, c in enumerate(all_chunks) if c.hierarchical_id == chunk.hierarchical_id), None)
                    
                    if chunk_index is None:
                        raise ValueError(f"Chunk {chunk.hierarchical_id} not found in all_chunks")
                    
                    # Get context for this chunk
                    context = self._build_chunk_context(chunk, all_chunks, chunk_index, toc_summary)
                    
                    # Generate content for chunk
                    content = self._generate_chunk_content(llm_client, chunk, context)
                    
                    # Generate summary of the content
                    summary = self._generate_chunk_summary(llm_client, chunk, content)
                    
                    # Update chunk in database
                    chunk.content = content
                    chunk.summary = summary
                    chunk.status = ChunkStatus.COMPLETED
                    db.save_chunks([chunk])
                    
                    # Save individual chunk output
                    self._save_chunk_output(chunk)
                    
                    processed_count += 1
                    self._persist_task_progress(chunk.hierarchical_id, f"ProcessChunksTask_Iteration_{self.iteration}", "COMPLETED")
                    print(f"✅ Processed chunk {chunk.hierarchical_id}: {chunk.title}")
                    
                except Exception as e:
                    chunk.status = ChunkStatus.FAILED
                    db.save_chunks([chunk])
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
        toc_short_path = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/toc_short.txt"
        with open(toc_short_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _build_chunk_context(self, chunk: ChunkData, all_chunks: list, current_index: int, toc_summary: str) -> dict:
        """Build context for chunk generation"""
        # Get previous chunk content
        previous_chunk = all_chunks[current_index - 1] if current_index > 0 else None
        
        # Get 5 previous summaries
        start_prev = max(0, current_index - 5)
        previous_summaries = [c.summary for c in all_chunks[start_prev:current_index] if c.summary]
        
        # Get 5 next summaries (only titles for now)
        end_next = min(len(all_chunks), current_index + 6)
        next_titles = [f"{c.hierarchical_id}: {c.title}" for c in all_chunks[current_index + 1:end_next]]
        
        return {
            "hierarchical_id": chunk.hierarchical_id,
            "title": chunk.title,
            "toc_summary": toc_summary,
            "previous_content": previous_chunk.content if previous_chunk else None,
            "previous_summaries": previous_summaries,
            "next_titles": next_titles
        }
    
    def _generate_chunk_content(self, llm_client: LLMClient, chunk: ChunkData, context: dict) -> str:
        """Generate content for chunk using LLM"""
        prompt_config = self.task_config['content_prompt']
        
        # Format context for prompt
        context_text = self._format_context_for_prompt(context)
        
        # Create prompt
        user_prompt = prompt_config['user'].format(
            hierarchical_id=context['hierarchical_id'],
            title=context['title'],
            context=context_text
        )
        
        full_prompt = f"{prompt_config['system']}\n\n{user_prompt}"
        
        return llm_client.complete(full_prompt)
    
    def _generate_chunk_summary(self, llm_client: LLMClient, chunk: ChunkData, content: str) -> str:
        """Generate summary of chunk content"""
        prompt_config = self.task_config['summary_prompt']
        
        user_prompt = prompt_config['user'].format(
            hierarchical_id=chunk.hierarchical_id,
            title=chunk.title,
            content=content
        )
        
        full_prompt = f"{prompt_config['system']}\n\n{user_prompt}"
        
        return llm_client.complete(full_prompt)
    
    def _format_context_for_prompt(self, context: dict) -> str:
        """Format context information for LLM prompt"""
        parts = [f"HIERARCHIA: {context['hierarchical_id']}"]
        
        if context['toc_summary']:
            parts.append(f"STRESZCZENIE CAŁOŚCI:\n{context['toc_summary']}")
        
        if context['previous_content']:
            parts.append(f"POPRZEDNI FRAGMENT:\n{context['previous_content']}")
        
        if context['previous_summaries']:
            summaries = '\n'.join(context['previous_summaries'])
            parts.append(f"POPRZEDNIE STRESZCZENIA:\n{summaries}")
        
        if context['next_titles']:
            titles = '\n'.join(context['next_titles'])
            parts.append(f"NASTĘPNE TEMATY:\n{titles}")
        
        return '\n\n'.join(parts)
    
    def _save_chunk_output(self, chunk: ChunkData):
        """Save individual chunk output to file"""
        chunk_dir = f"{self.output_dir}/chunks"
        Path(chunk_dir).mkdir(parents=True, exist_ok=True)
        
        chunk_file = f"{chunk_dir}/{chunk.hierarchical_id.replace('.', '_')}.txt"
        with open(chunk_file, 'w', encoding='utf-8') as f:
            f.write(f"ID: {chunk.hierarchical_id}\n")
            f.write(f"TITLE: {chunk.title}\n")
            f.write(f"STATUS: {chunk.status.value}\n")
            f.write(f"ITERATION: {self.iteration}\n")
            f.write(f"\nCONTENT:\n{chunk.content}\n")
            f.write(f"\nSUMMARY:\n{chunk.summary}\n")
    
    def _persist_task_progress(self, hierarchical_id: str, task_name: str, status: str):
        """Persist task progress to file"""
        progress_file = self.config.get_progress_file()
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | {hierarchical_id} | {task_name} | {status}\n")