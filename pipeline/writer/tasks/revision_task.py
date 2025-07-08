import luigi
from pathlib import Path
from typing import List, Optional

from components.structured_task import StructuredTask
from pipeline.writer.database import DatabaseManager
from pipeline.writer.models import ChunkStatus, ChunkData
from pipeline.writer.config_loader import ConfigLoader
from llm import LLMClient


class RevisionTask(StructuredTask):
    toc_path = luigi.Parameter()
    iteration = luigi.IntParameter(default=1)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.task_config = self.config.get_task_config('RevisionTask')
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/revision/iteration_{self.iteration}"
    
    def requires(self):
        from .quality_check_task import QualityCheckTask
        return QualityCheckTask(
            toc_path=self.toc_path,
            iteration=self.iteration
        )
    
    def output(self):
        return luigi.LocalTarget(f"{self.output_dir}/completed.flag")
    
    def run(self):
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Persist task start
        self._persist_task_progress("GLOBAL", f"RevisionTask_Iteration_{self.iteration}", "STARTED")
        
        try:
            # Initialize components
            db = DatabaseManager(self.config.get_database_path())
            llm_client = LLMClient(model=self.task_config['model'])
            
            # Find chunks that need revision from this iteration
            chunks_to_revise = self._find_chunks_needing_revision_from_iteration()
            
            if not chunks_to_revise:
                print(f"✅ No chunks need revision in iteration {self.iteration}")
                with self.output().open('w') as f:
                    f.write("No chunks required revision")
                self._persist_task_progress("GLOBAL", f"RevisionTask_Iteration_{self.iteration}", "COMPLETED")
                return
            
            # Get all chunks for context
            all_chunks = db.get_chunks_by_status(ChunkStatus.COMPLETED)
            all_chunks.sort(key=lambda x: x.hierarchical_id)
            
            # Load TOC summary
            toc_summary = self._load_toc_summary()
            
            # Process each chunk needing revision
            revised_count = 0
            for chunk_id in chunks_to_revise:
                # Find the chunk and its neighbors
                target_chunk = self._find_chunk_by_hierarchical_id(all_chunks, chunk_id)
                if not target_chunk:
                    print(f"⚠️ Chunk {chunk_id} not found")
                    continue
                
                neighbors = self._get_neighboring_chunks(all_chunks, target_chunk)
                
                self._persist_task_progress(chunk_id, f"RevisionTask_Iteration_{self.iteration}", "IN_PROGRESS")
                
                try:
                    # Store original content for comparison
                    original_content = target_chunk.content
                    original_summary = target_chunk.summary
                    
                    # Revise the chunk
                    revised_content = self._revise_chunk(llm_client, target_chunk, neighbors, toc_summary)
                    revised_summary = self._generate_revised_summary(llm_client, target_chunk, revised_content)
                    
                    # Update chunk in database
                    target_chunk.content = revised_content
                    target_chunk.summary = revised_summary
                    target_chunk.status = ChunkStatus.COMPLETED
                    db.save_chunks([target_chunk])
                    
                    # Save revision output with before/after comparison
                    self._save_revision_output(target_chunk, neighbors, original_content, original_summary)
                    
                    revised_count += 1
                    self._persist_task_progress(chunk_id, f"RevisionTask_Iteration_{self.iteration}", "COMPLETED")
                    print(f"✅ Revised chunk {chunk_id}: {target_chunk.title}")
                    
                except Exception as e:
                    target_chunk.status = ChunkStatus.FAILED
                    db.save_chunks([target_chunk])
                    self._persist_task_progress(chunk_id, f"RevisionTask_Iteration_{self.iteration}", "FAILED")
                    print(f"❌ Failed to revise chunk {chunk_id}: {str(e)}")
            
            # Create revision summary report
            self._create_revision_summary(chunks_to_revise, revised_count)
            
            # Create completion flag
            with self.output().open('w') as f:
                f.write(f"Revision iteration {self.iteration} completed: {revised_count}/{len(chunks_to_revise)} chunks revised")
            
            # Persist task completion
            self._persist_task_progress("GLOBAL", f"RevisionTask_Iteration_{self.iteration}", "COMPLETED")
            
            print(f"✅ Completed revision iteration {self.iteration}: {revised_count}/{len(chunks_to_revise)} chunks")
            
        except Exception as e:
            self._persist_task_progress("GLOBAL", f"RevisionTask_Iteration_{self.iteration}", "FAILED")
            raise
    
    def _find_chunks_needing_revision_from_iteration(self) -> List[str]:
        """Find chunks that need revision from current iteration quality check"""
        chunks_to_revise = []
        progress_file = self.config.get_progress_file()
        
        if not Path(progress_file).exists():
            return chunks_to_revise
        
        # Look for NEEDS_REWRITE or NEEDS_REVIEW from QualityCheckTask in current iteration
        with open(progress_file, 'r') as f:
            for line in f:
                if ('QualityCheckTask' in line and 
                    ('NEEDS_REWRITE' in line or 'NEEDS_REVIEW' in line)):
                    
                    # Check if this is from a recent quality check (could be from any iteration)
                    parts = line.strip().split(' | ')
                    if len(parts) >= 2:
                        hierarchical_id = parts[1]
                        if hierarchical_id not in chunks_to_revise and hierarchical_id != "GLOBAL":
                            chunks_to_revise.append(hierarchical_id)
        
        return chunks_to_revise
    
    def _find_chunk_by_hierarchical_id(self, chunks: List[ChunkData], hierarchical_id: str) -> Optional[ChunkData]:
        """Find chunk by hierarchical ID"""
        for chunk in chunks:
            if chunk.hierarchical_id == hierarchical_id:
                return chunk
        return None
    
    def _get_neighboring_chunks(self, all_chunks: List[ChunkData], target_chunk: ChunkData) -> dict:
        """Get previous and next chunks"""
        target_index = None
        for i, chunk in enumerate(all_chunks):
            if chunk.hierarchical_id == target_chunk.hierarchical_id:
                target_index = i
                break
        
        if target_index is None:
            return {"previous": None, "next": None}
        
        previous_chunk = all_chunks[target_index - 1] if target_index > 0 else None
        next_chunk = all_chunks[target_index + 1] if target_index < len(all_chunks) - 1 else None
        
        return {
            "previous": previous_chunk,
            "next": next_chunk
        }
    
    def _revise_chunk(self, llm_client: LLMClient, chunk: ChunkData, neighbors: dict, toc_summary: str) -> str:
        """Revise chunk content based on TOC summary and neighbors"""
        # Build context from neighbors and TOC
        context = self._build_revision_context(chunk, neighbors, toc_summary)
        
        # Create revision prompt
        prompt = self._create_revision_prompt(chunk, context)
        
        return llm_client.complete(prompt)
    
    def _build_revision_context(self, chunk: ChunkData, neighbors: dict, toc_summary: str) -> str:
        """Build context for revision"""
        context_parts = [f"SCENARIUSZ CAŁOŚCI:\n{toc_summary}"]
        
        if neighbors["previous"]:
            context_parts.append(f"POPRZEDNI FRAGMENT ({neighbors['previous'].hierarchical_id}):\n{neighbors['previous'].summary}")
        
        if neighbors["next"]:
            context_parts.append(f"NASTĘPNY FRAGMENT ({neighbors['next'].hierarchical_id}):\n{neighbors['next'].summary}")
        
        return "\n\n".join(context_parts)
    
    def _create_revision_prompt(self, chunk: ChunkData, context: str) -> str:
        """Create revision prompt"""
        prompt_config = self.task_config['prompt']
        
        user_prompt = prompt_config['user'].format(
            hierarchical_id=chunk.hierarchical_id,
            title=chunk.title,
            context=context,
            original_content=chunk.content
        )
        
        return f"{prompt_config['system']}\n\n{user_prompt}"
    
    def _generate_revised_summary(self, llm_client: LLMClient, chunk: ChunkData, content: str) -> str:
        """Generate summary of revised content"""
        prompt_config = self.task_config['summary_prompt']
        
        user_prompt = prompt_config['user'].format(
            hierarchical_id=chunk.hierarchical_id,
            title=chunk.title,
            content=content
        )
        
        return llm_client.complete(f"{prompt_config['system']}\n\n{user_prompt}")
    
    def _save_revision_output(self, chunk: ChunkData, neighbors: dict, original_content: str, original_summary: str):
        """Save revision output to file with before/after comparison"""
        revision_dir = f"{self.output_dir}/revisions"
        Path(revision_dir).mkdir(parents=True, exist_ok=True)
        
        revision_file = f"{revision_dir}/{chunk.hierarchical_id.replace('.', '_')}_revision_iter_{self.iteration}.txt"
        with open(revision_file, 'w', encoding='utf-8') as f:
            f.write(f"CHUNK REVISION - ITERATION {self.iteration}\n")
            f.write(f"ID: {chunk.hierarchical_id}\n")
            f.write(f"TITLE: {chunk.title}\n\n")
            
            if neighbors["previous"]:
                f.write(f"PREVIOUS: {neighbors['previous'].hierarchical_id} - {neighbors['previous'].title}\n")
            if neighbors["next"]:
                f.write(f"NEXT: {neighbors['next'].hierarchical_id} - {neighbors['next'].title}\n")
            
            f.write(f"\n" + "="*50 + "\n")
            f.write(f"ORIGINAL CONTENT:\n{original_content}\n")
            f.write(f"\nORIGINAL SUMMARY:\n{original_summary}\n")
            
            f.write(f"\n" + "="*50 + "\n")
            f.write(f"REVISED CONTENT:\n{chunk.content}\n")
            f.write(f"\nREVISED SUMMARY:\n{chunk.summary}\n")
    
    def _create_revision_summary(self, chunks_to_revise: List[str], revised_count: int):
        """Create summary of revision iteration"""
        summary_file = f"{self.output_dir}/revision_iteration_{self.iteration}_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"REVISION SUMMARY - ITERATION {self.iteration}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Chunks identified for revision: {len(chunks_to_revise)}\n")
            f.write(f"Chunks successfully revised: {revised_count}\n")
            f.write(f"Success rate: {(revised_count/len(chunks_to_revise)*100):.1f}%\n\n")
            
            f.write("CHUNKS TO REVISE:\n")
            for chunk_id in chunks_to_revise:
                f.write(f"- {chunk_id}\n")
            
            if revised_count == len(chunks_to_revise):
                f.write(f"\n✅ All chunks successfully revised in iteration {self.iteration}")
            else:
                failed_count = len(chunks_to_revise) - revised_count
                f.write(f"\n⚠️ {failed_count} chunks failed revision in iteration {self.iteration}")
    
    def _load_toc_summary(self) -> str:
        """Load TOC summary for context"""
        toc_short_path = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/toc_short.txt"
        with open(toc_short_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _persist_task_progress(self, hierarchical_id: str, task_name: str, status: str):
        """Persist task progress to file"""
        progress_file = self.config.get_progress_file()
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | {hierarchical_id} | {task_name} | {status}\n")