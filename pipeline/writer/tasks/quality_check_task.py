import luigi
from pathlib import Path
from typing import List

from pipeline.writer.tasks.base_writer_task import BaseWriterTask
from writer.writer_service import WriterService
from writer.models import ChunkStatus, ChunkData
from pipeline.writer.config_loader import ConfigLoader
from llm import LLMClient


class QualityCheckTask(BaseWriterTask):
    toc_path = luigi.Parameter()
    iteration = luigi.IntParameter(default=1)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.task_config = self.config.get_task_config('QualityCheckTask')
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/quality_check/iteration_{self.iteration}"
        self.batch_size = 5
    
    @property 
    def task_name(self) -> str:
        return "quality_check"
    
    def requires(self):
        from .process_chunks_task import ProcessChunksTask
        return ProcessChunksTask(
            toc_path=self.toc_path,
            iteration=self.iteration
        )
    
    def output(self):
        return luigi.LocalTarget(f"{self.output_dir}/completed.flag")
    
    def run(self):
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Persist task start
        self._persist_task_progress("GLOBAL", f"QualityCheckTask_Iteration_{self.iteration}", "STARTED")
        
        try:
            # Initialize WriterService and LLM
            writer_service = WriterService(storage_dir="output/writer_storage")
            llm_client = LLMClient(model=self.task_config['model'])
            
            # Get recently completed chunks (from this iteration)
            recently_completed_chunks = self._get_recently_completed_chunks(writer_service)
            
            if not recently_completed_chunks:
                print(f"✅ No recently completed chunks to check in iteration {self.iteration}")
                with open(self.output().path, 'w', encoding='utf-8') as f:
                    f.write("No chunks to check")
                self._persist_task_progress("GLOBAL", f"QualityCheckTask_Iteration_{self.iteration}", "COMPLETED")
                return
            
            # Load TOC summary for context
            toc_summary = self._load_toc_summary()
            
            # Process chunks in batches
            total_issues = 0
            batch_count = 0
            
            for i in range(0, len(recently_completed_chunks), self.batch_size):
                batch = recently_completed_chunks[i:i + self.batch_size]
                batch_count += 1
                
                self._persist_task_progress(f"BATCH_{batch_count}_ITER_{self.iteration}", "QualityCheckTask", "IN_PROGRESS")
                
                # Check quality of this batch
                issues = self._check_batch_quality(llm_client, writer_service, batch, toc_summary)
                
                if issues:
                    # Save issues to file
                    self._save_batch_issues(batch_count, batch, issues)
                    total_issues += len(issues)
                    
                    # Mark problematic chunks for revision
                    self._mark_chunks_for_revision(writer_service, issues)
                    
                    print(f"⚠️ Iteration {self.iteration}, Batch {batch_count}: Found {len(issues)} issues")
                else:
                    print(f"✅ Iteration {self.iteration}, Batch {batch_count}: No issues found")
                
                self._persist_task_progress(f"BATCH_{batch_count}_ITER_{self.iteration}", "QualityCheckTask", "COMPLETED")
            
            # Create quality report for this iteration
            self._create_iteration_quality_report(recently_completed_chunks, total_issues, batch_count)
            
            # Create completion flag
            with open(self.output().path, 'w', encoding='utf-8') as f:
                f.write(f"Quality check iteration {self.iteration} completed: {total_issues} issues found in {batch_count} batches")
            
            # Persist task completion
            self._persist_task_progress("GLOBAL", f"QualityCheckTask_Iteration_{self.iteration}", "COMPLETED")
            
            print(f"✅ Quality check iteration {self.iteration} completed: {total_issues} total issues found")
            
        except Exception as e:
            self._persist_task_progress("GLOBAL", f"QualityCheckTask_Iteration_{self.iteration}", "FAILED")
            raise
    
    def _get_recently_completed_chunks(self, writer_service: WriterService) -> List[ChunkData]:
        """Get chunks that were recently completed (in current processing cycle)"""
        # Read progress file to find chunks completed in this iteration
        progress_file = self.config.get_progress_file()
        recently_completed_ids = set()
        
        if Path(progress_file).exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if f"ProcessChunksTask_Iteration_{self.iteration}" in line and "COMPLETED" in line:
                        parts = line.strip().split(' | ')
                        if len(parts) >= 2:
                            hierarchical_id = parts[1]
                            if hierarchical_id != "GLOBAL":
                                recently_completed_ids.add(hierarchical_id)
        
        # Get these chunks from WriterService
        all_completed = writer_service.get_chunks_by_status(ChunkStatus.COMPLETED)
        recently_completed = [c for c in all_completed if c.hierarchical_id in recently_completed_ids]
        recently_completed.sort(key=lambda x: x.hierarchical_id)
        
        return recently_completed
    
    def _check_batch_quality(self, llm_client: LLMClient, writer_service: WriterService, 
                           batch: List[ChunkData], toc_summary: str) -> List[dict]:
        """Check quality of a batch of chunks with semantic context"""
        # Prepare batch content for analysis
        batch_content = self._format_batch_for_analysis(batch)
        
        # Get semantic context for quality analysis
        semantic_context = self._get_semantic_context_for_batch(writer_service, batch)
        
        # Create quality check prompt
        prompt = self._create_quality_prompt(batch_content, toc_summary, semantic_context)
        
        # Get LLM analysis
        print(f"🔍 Analyzing quality of batch with {len(batch)} chunks...")
        analysis = llm_client.chat(prompt)
        
        # Parse issues from analysis
        issues = self._parse_quality_issues(analysis, batch)
        
        return issues
    
    def _get_semantic_context_for_batch(self, writer_service: WriterService, batch: List[ChunkData]) -> str:
        """Get semantic context using WriterService search capabilities"""
        context_parts = []
        
        # For each chunk in batch, find semantically similar chunks for cross-reference
        for chunk in batch:
            if chunk.content:
                # Find similar chunks to check for consistency
                similar_results = writer_service.search_chunks_semantic(
                    query=chunk.content[:200],  # First 200 chars as query
                    max_results=3,
                    min_similarity=0.7
                )
                
                if similar_results:
                    similar_chunks = [f"{r.chunk.hierarchical_id}: {r.chunk.title}" 
                                    for r in similar_results if r.chunk.id != chunk.id]
                    if similar_chunks:
                        context_parts.append(f"Chunks similar to {chunk.hierarchical_id}: {', '.join(similar_chunks)}")
        
        return "\n".join(context_parts) if context_parts else "No semantic similarities found"
    
    def _format_batch_for_analysis(self, batch: List[ChunkData]) -> str:
        """Format batch chunks for quality analysis"""
        formatted_chunks = []
        
        for chunk in batch:
            chunk_text = f"""
CHUNK {chunk.hierarchical_id}: {chunk.title}
LEVEL: {chunk.level}
PARENT: {chunk.parent_hierarchical_id or 'None'}
STRESZCZENIE: {chunk.summary or 'Brak streszczenia'}
TREŚĆ: {chunk.content or 'Brak treści'}
---"""
            formatted_chunks.append(chunk_text)
        
        return "\n".join(formatted_chunks)
    
    def _create_quality_prompt(self, batch_content: str, toc_summary: str, semantic_context: str) -> str:
        """Create quality check prompt with semantic context"""
        prompt_config = self.task_config['prompt']
        
        # Format system prompt with parameters
        system_message = self.format_prompt_template(prompt_config['system'])
        
        # Format user prompt with parameters
        user_prompt = self.format_prompt_template(
            prompt_config['user'],
            toc_summary=toc_summary,
            batch_content=batch_content
        )
        
        # Add semantic context
        enhanced_prompt = f"{user_prompt}\n\nKONTEKST SEMANTYCZNY:\n{semantic_context}"
        
        # Add custom prompt if provided
        if self.custom_prompt.strip():
            enhanced_prompt = f"{enhanced_prompt}\n\nDODATKOWE INSTRUKCJE:\n{self.custom_prompt}"
        
        return f"{system_message}\n\n{enhanced_prompt}"
    
    def _parse_quality_issues(self, analysis: str, batch: List[ChunkData]) -> List[dict]:
        """Parse quality issues from LLM analysis"""
        issues = []
        
        # Simple parsing - look for hierarchical IDs mentioned as problematic
        lines = analysis.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for hierarchical IDs in the analysis
            for chunk in batch:
                if chunk.hierarchical_id in line and any(keyword in line.lower() for keyword in 
                    ['problem', 'błąd', 'niespójn', 'przepis', 'poprawi', 'zmieni', 'usuń', 'dodaj']):
                    
                    # Determine action type
                    action = 'review'
                    if any(word in line.lower() for word in ['przepis', 'rewrite']):
                        action = 'rewrite'
                    elif any(word in line.lower() for word in ['usuń', 'remove']):
                        action = 'remove'
                    elif any(word in line.lower() for word in ['dodaj', 'add']):
                        action = 'expand'
                    
                    issue = {
                        'hierarchical_id': chunk.hierarchical_id,
                        'chunk_id': chunk.id,
                        'issue_description': line,
                        'action': action,
                        'severity': self._assess_severity(line)
                    }
                    issues.append(issue)
        
        return issues
    
    def _assess_severity(self, line: str) -> str:
        """Assess issue severity based on keywords"""
        line_lower = line.lower()
        
        if any(word in line_lower for word in ['krytyczny', 'poważny', 'błąd', 'sprzeczność']):
            return 'high'
        elif any(word in line_lower for word in ['drobny', 'minor', 'lekki']):
            return 'low'
        else:
            return 'medium'
    
    def _save_batch_issues(self, batch_number: int, batch: List[ChunkData], issues: List[dict]):
        """Save issues found in batch to file"""
        issues_file = f"{self.output_dir}/iteration_{self.iteration}_batch_{batch_number}_issues.txt"
        
        with open(issues_file, 'w', encoding='utf-8') as f:
            f.write(f"QUALITY ISSUES - ITERATION {self.iteration}, BATCH {batch_number}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("BATCH CHUNKS:\n")
            for chunk in batch:
                f.write(f"- {chunk.hierarchical_id}: {chunk.title}\n")
            
            f.write(f"\nISSUES FOUND ({len(issues)}):\n")
            for issue in issues:
                f.write(f"\n- ID: {issue['hierarchical_id']}\n")
                f.write(f"  ACTION: {issue['action']}\n")
                f.write(f"  SEVERITY: {issue['severity']}\n")
                f.write(f"  DESCRIPTION: {issue['issue_description']}\n")
    
    def _mark_chunks_for_revision(self, writer_service: WriterService, issues: List[dict]):
        """Mark chunks with issues for revision"""
        for issue in issues:
            hierarchical_id = issue['hierarchical_id']
            action = issue['action']
            severity = issue['severity']
            
            self._persist_task_progress(hierarchical_id, "QualityCheckTask", f"NEEDS_{action.upper()}")
            
            # Optionally update chunk status for high severity issues
            if severity == 'high':
                chunk = writer_service.get_chunk_by_hierarchical_id(hierarchical_id)
                if chunk:
                    chunk.status = ChunkStatus.IN_PROGRESS  # Mark for revision
                    writer_service.persistence.save_chunk(chunk)
                    writer_service.chunks[chunk.id] = chunk
    
    def _create_iteration_quality_report(self, checked_chunks: List[ChunkData], total_issues: int, batch_count: int):
        """Create quality report for this iteration"""
        report_file = f"{self.output_dir}/iteration_{self.iteration}_quality_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"QUALITY CHECK REPORT - ITERATION {self.iteration}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Chunks checked: {len(checked_chunks)}\n")
            f.write(f"Batches processed: {batch_count}\n")
            f.write(f"Issues found: {total_issues}\n")
            
            if len(checked_chunks) > 0:
                quality_score = ((len(checked_chunks) - total_issues) / len(checked_chunks) * 100)
                f.write(f"Quality score: {quality_score:.1f}%\n\n")
            
            if total_issues == 0:
                f.write("✅ No quality issues found in this iteration.\n")
            else:
                f.write(f"⚠️ {total_issues} issues require revision.\n")
                f.write("Check individual batch files for details.\n")
            
            f.write(f"\nCHUNKS CHECKED:\n")
            for chunk in checked_chunks:
                f.write(f"- {chunk.hierarchical_id}: {chunk.title}\n")
    
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