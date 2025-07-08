import luigi
from pathlib import Path
from typing import List, Optional


from pipeline.writer.tasks.base_writer_task import BaseWriterTask
from writer.writer_service import WriterService
from writer.models import ChunkStatus, ChunkData
from pipeline.writer.config_loader import ConfigLoader
from llm import LLMClient


class RevisionTask(BaseWriterTask):
    toc_path = luigi.Parameter()
    iteration = luigi.IntParameter(default=1)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.task_config = self.config.get_task_config('RevisionTask')
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/revision/iteration_{self.iteration}"
    
    @property 
    def task_name(self) -> str:
        return "revistion_task"
    
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
            # Initialize WriterService and LLM
            writer_service = WriterService(storage_dir="output/writer_storage")
            llm_client = LLMClient(model=self.task_config['model'])
            
            # Find chunks that need revision from THIS SPECIFIC iteration
            chunks_to_revise = self._find_chunks_needing_revision_from_current_iteration()
            
            if not chunks_to_revise:
                print(f"✅ No chunks need revision in iteration {self.iteration}")
                with open(self.output().path, 'w', encoding='utf-8') as f:
                    f.write(f"No chunks required revision in iteration {self.iteration}")
                self._persist_task_progress("GLOBAL", f"RevisionTask_Iteration_{self.iteration}", "COMPLETED")
                return
            
            print(f"🔧 Found {len(chunks_to_revise)} chunks needing revision from iteration {self.iteration}: {chunks_to_revise}")
            
            # Load TOC summary
            toc_summary = self._load_toc_summary()
            
            # Process each chunk needing revision
            revised_count = 0
            for hierarchical_id in chunks_to_revise:
                # Find the chunk
                target_chunk = writer_service.get_chunk_by_hierarchical_id(hierarchical_id)
                if not target_chunk:
                    print(f"⚠️ Chunk {hierarchical_id} not found")
                    continue
                
                self._persist_task_progress(hierarchical_id, f"RevisionTask_Iteration_{self.iteration}", "IN_PROGRESS")
                
                try:
                    # Store original content for comparison
                    original_content = target_chunk.content
                    original_summary = target_chunk.summary
                    
                    # Get comprehensive context for revision
                    revision_context = self._get_revision_context(writer_service, target_chunk, toc_summary)
                    
                    # Revise the chunk
                    revised_content = self._revise_chunk(llm_client, target_chunk, revision_context)
                    revised_summary = self._generate_revised_summary(llm_client, target_chunk, revised_content)
                    
                    # Update chunk using WriterService (handles embedding update automatically)
                    success = writer_service.update_chunk_content(
                        chunk_id=target_chunk.id,
                        content=revised_content,
                        summary=revised_summary
                    )
                    
                    if success:
                        # Save revision output with before/after comparison
                        self._save_revision_output(
                            target_chunk, 
                            revision_context, 
                            original_content, 
                            original_summary,
                            revised_content,
                            revised_summary
                        )
                        
                        revised_count += 1
                        self._persist_task_progress(hierarchical_id, f"RevisionTask_Iteration_{self.iteration}", "COMPLETED")
                        print(f"✅ Revised chunk {hierarchical_id}: {target_chunk.title}")
                    else:
                        raise ValueError("Failed to update chunk in WriterService")
                    
                except Exception as e:
                    # Mark chunk as failed
                    target_chunk.status = ChunkStatus.FAILED
                    writer_service.persistence.save_chunk(target_chunk)
                    writer_service.chunks[target_chunk.id] = target_chunk
                    
                    self._persist_task_progress(hierarchical_id, f"RevisionTask_Iteration_{self.iteration}", "FAILED")
                    print(f"❌ Failed to revise chunk {hierarchical_id}: {str(e)}")
            
            # Create revision summary report
            self._create_revision_summary(chunks_to_revise, revised_count)
            
            # Create completion flag
            with open(self.output().path, 'w', encoding='utf-8') as f:
                f.write(f"Revision iteration {self.iteration} completed: {revised_count}/{len(chunks_to_revise)} chunks revised")
            
            # Persist task completion
            self._persist_task_progress("GLOBAL", f"RevisionTask_Iteration_{self.iteration}", "COMPLETED")
            
            print(f"✅ Completed revision iteration {self.iteration}: {revised_count}/{len(chunks_to_revise)} chunks")
            
        except Exception as e:
            self._persist_task_progress("GLOBAL", f"RevisionTask_Iteration_{self.iteration}", "FAILED")
            raise
    
    def _find_chunks_needing_revision_from_current_iteration(self) -> List[str]:
        """Find chunks that need revision from CURRENT iteration ONLY"""
        chunks_to_revise = []
        progress_file = self.config.get_progress_file()
        
        if not Path(progress_file).exists():
            print(f"📝 No progress file found at {progress_file}")
            return chunks_to_revise
        
        # Look for NEEDS_REWRITE or NEEDS_REVIEW from QualityCheckTask in CURRENT iteration ONLY
        target_quality_check = f"QualityCheckTask_Iteration_{self.iteration}"
        
        print(f"🔍 Looking for revision needs from: {target_quality_check}")
        
        with open(progress_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Must match EXACTLY this iteration's quality check
                if (target_quality_check in line and 
                    ('NEEDS_REWRITE' in line or 'NEEDS_REVIEW' in line or 'NEEDS_EXPAND' in line)):
                    
                    parts = line.strip().split(' | ')
                    if len(parts) >= 2:
                        hierarchical_id = parts[1]
                        if hierarchical_id not in chunks_to_revise and hierarchical_id != "GLOBAL":
                            chunks_to_revise.append(hierarchical_id)
                            print(f"🔧 Found chunk needing revision: {hierarchical_id}")
        
        print(f"📝 Total chunks to revise from iteration {self.iteration}: {len(chunks_to_revise)}")
        return chunks_to_revise
    
    def _get_revision_context(self, writer_service: WriterService, target_chunk: ChunkData, toc_summary: str) -> dict:
        """Get comprehensive context for revision using WriterService capabilities"""
        # Get basic processing context (previous/next chunks)
        basic_context = writer_service.get_processing_context(target_chunk)
        
        # Get semantic context - similar chunks for reference
        semantic_context = []
        if target_chunk.content:
            similar_results = writer_service.search_chunks_semantic(
                query=target_chunk.content[:300],  # First 300 chars
                max_results=5,
                min_similarity=0.6
            )
            
            semantic_context = [
                {
                    'hierarchical_id': r.chunk.hierarchical_id,
                    'title': r.chunk.title,
                    'summary': r.chunk.summary,
                    'similarity': r.similarity_score
                }
                for r in similar_results if r.chunk.id != target_chunk.id
            ]
        
        # Get contextual chunks for LLM processing
        contextual_chunks = writer_service.get_contextual_chunks(
            query=f"{target_chunk.hierarchical_id} {target_chunk.title}",
            max_chunks=8,
            threshold=0.5
        )
        
        return {
            'basic_context': basic_context,
            'semantic_context': semantic_context,
            'contextual_chunks': contextual_chunks,
            'toc_summary': toc_summary,
            'target_chunk': target_chunk
        }
    
    def _revise_chunk(self, llm_client: LLMClient, chunk: ChunkData, revision_context: dict) -> str:
        """Revise chunk content using comprehensive context"""
        # Build context from all sources
        context_text = self._build_revision_context_text(revision_context)
        
        # Create revision prompt
        prompt = self._create_revision_prompt(chunk, context_text)
        
        print(f"🔧 Revising chunk {chunk.hierarchical_id} with semantic context")
        return llm_client.chat(prompt)
    
    def _build_revision_context_text(self, revision_context: dict) -> str:
        """Build comprehensive context text for revision"""
        context_parts = []
        
        # TOC summary
        context_parts.append(f"SCENARIUSZ CAŁOŚCI:\n{revision_context['toc_summary']}")
        
        # Basic context (previous/next)
        basic = revision_context['basic_context']
        if basic.get('previous_chunk') and basic['previous_chunk'].summary:
            context_parts.append(f"POPRZEDNI FRAGMENT ({basic['previous_chunk'].hierarchical_id}):\n{basic['previous_chunk'].summary}")
        
        if basic.get('next_chunk') and basic['next_chunk'].summary:
            context_parts.append(f"NASTĘPNY FRAGMENT ({basic['next_chunk'].hierarchical_id}):\n{basic['next_chunk'].summary}")
        
        # Semantic context
        if revision_context['semantic_context']:
            similar_parts = []
            for similar in revision_context['semantic_context'][:3]:  # Top 3 similar
                similar_parts.append(f"- {similar['hierarchical_id']}: {similar['title']} (podobieństwo: {similar['similarity']:.2f})")
            context_parts.append(f"PODOBNE FRAGMENTY:\n" + "\n".join(similar_parts))
        
        # Contextual chunks
        if revision_context['contextual_chunks']:
            contextual_parts = []
            for ctx in revision_context['contextual_chunks'][:3]:  # Top 3 contextual
                contextual_parts.append(f"- {ctx['hierarchical_id']}: {ctx['title']}")
            context_parts.append(f"KONTEKST TEMATYCZNY:\n" + "\n".join(contextual_parts))
        
        return "\n\n".join(context_parts)
    
    def _create_revision_prompt(self, chunk: ChunkData, context: str) -> str:
        """Create revision prompt with comprehensive context"""
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
        
        return llm_client.chat(f"{prompt_config['system']}\n\n{user_prompt}")
    
    def _save_revision_output(self, chunk: ChunkData, revision_context: dict, 
                            original_content: str, original_summary: str,
                            revised_content: str, revised_summary: str):
        """Save comprehensive revision output with context analysis"""
        revision_dir = f"{self.output_dir}/revisions"
        Path(revision_dir).mkdir(parents=True, exist_ok=True)
        
        revision_file = f"{revision_dir}/{chunk.hierarchical_id.replace('.', '_')}_revision_iter_{self.iteration}.txt"
        with open(revision_file, 'w', encoding='utf-8') as f:
            f.write(f"CHUNK REVISION - ITERATION {self.iteration}\n")
            f.write(f"ID: {chunk.hierarchical_id}\n")
            f.write(f"TITLE: {chunk.title}\n")
            f.write(f"LEVEL: {chunk.level}\n")
            f.write(f"PARENT: {chunk.parent_hierarchical_id or 'None'}\n\n")
            
            # Context information
            basic = revision_context['basic_context']
            if basic.get('previous_chunk'):
                f.write(f"PREVIOUS: {basic['previous_chunk'].hierarchical_id} - {basic['previous_chunk'].title}\n")
            if basic.get('next_chunk'):
                f.write(f"NEXT: {basic['next_chunk'].hierarchical_id} - {basic['next_chunk'].title}\n")
            
            # Semantic context
            if revision_context['semantic_context']:
                f.write(f"\nSEMANTIC SIMILAR CHUNKS:\n")
                for similar in revision_context['semantic_context']:
                    f.write(f"- {similar['hierarchical_id']}: {similar['title']} ({similar['similarity']:.2f})\n")
            
            f.write(f"\n" + "="*50 + "\n")
            f.write(f"ORIGINAL CONTENT:\n{original_content}\n")
            f.write(f"\nORIGINAL SUMMARY:\n{original_summary}\n")
            
            f.write(f"\n" + "="*50 + "\n")
            f.write(f"REVISED CONTENT:\n{revised_content}\n")
            f.write(f"\nREVISED SUMMARY:\n{revised_summary}\n")
    
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
        toc_short_path = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/{self.config.get_output_config()['toc_short_filename']}"
        
        try:
            with open(toc_short_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"⚠️ TOC summary not found at {toc_short_path}")
            return "TOC summary not available"
    
    def _persist_task_progress(self, hierarchical_id: str, task_name: str, status: str):
        """Persist task progress to file"""
        progress_file = self.config.get_progress_file()
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'a', encoding='utf-8') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | {hierarchical_id} | {task_name} | {status}\n")