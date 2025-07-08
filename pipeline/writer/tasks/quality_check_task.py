import luigi
from pathlib import Path
from typing import List

from components.structured_task import StructuredTask
from pipeline.writer.database import DatabaseManager
from pipeline.writer.models import ChunkStatus, ChunkData
from pipeline.writer.config_loader import ConfigLoader
from llm import LLMClient


class QualityCheckTask(StructuredTask):
    toc_path = luigi.Parameter()
    iteration = luigi.IntParameter(default=1)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.task_config = self.config.get_task_config('QualityCheckTask')
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/quality_check/iteration_{self.iteration}"
        self.batch_size = 5
    
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
            # Initialize components
            db = DatabaseManager(self.config.get_database_path())
            llm_client = LLMClient(model=self.task_config['model'])
            
            # Get recently completed chunks (from this iteration)
            recently_completed_chunks = self._get_recently_completed_chunks(db)
            
            if not recently_completed_chunks:
                print(f"✅ No recently completed chunks to check in iteration {self.iteration}")
                with self.output().open('w') as f:
                    f.write("No chunks to check")
                self._persist_task_progress("GLOBAL", f"QualityCheckTask_Iteration_{self.iteration}", "COMPLETED")
                return
            
            # Load TOC summary for context
            toc_summary = self._load_toc_summary()
            
            # Process chunks in batches of 5
            total_issues = 0
            batch_count = 0
            
            for i in range(0, len(recently_completed_chunks), self.batch_size):
                batch = recently_completed_chunks[i:i + self.batch_size]
                batch_count += 1
                
                self._persist_task_progress(f"BATCH_{batch_count}_ITER_{self.iteration}", "QualityCheckTask", "IN_PROGRESS")
                
                # Check quality of this batch
                issues = self._check_batch_quality(llm_client, batch, toc_summary)
                
                if issues:
                    # Save issues to file
                    self._save_batch_issues(batch_count, batch, issues)
                    total_issues += len(issues)
                    
                    # Mark problematic chunks for revision
                    self._mark_chunks_for_revision(db, issues)
                    
                    print(f"⚠️ Iteration {self.iteration}, Batch {batch_count}: Found {len(issues)} issues")
                else:
                    print(f"✅ Iteration {self.iteration}, Batch {batch_count}: No issues found")
                
                self._persist_task_progress(f"BATCH_{batch_count}_ITER_{self.iteration}", "QualityCheckTask", "COMPLETED")
            
            # Create quality report for this iteration
            self._create_iteration_quality_report(recently_completed_chunks, total_issues, batch_count)
            
            # Create completion flag
            with self.output().open('w') as f:
                f.write(f"Quality check iteration {self.iteration} completed: {total_issues} issues found in {batch_count} batches")
            
            # Persist task completion
            self._persist_task_progress("GLOBAL", f"QualityCheckTask_Iteration_{self.iteration}", "COMPLETED")
            
            print(f"✅ Quality check iteration {self.iteration} completed: {total_issues} total issues found")
            
        except Exception as e:
            self._persist_task_progress("GLOBAL", f"QualityCheckTask_Iteration_{self.iteration}", "FAILED")
            raise
    
    def _get_recently_completed_chunks(self, db: DatabaseManager) -> List[ChunkData]:
        """Get chunks that were recently completed (in current processing cycle)"""
        # Read progress file to find chunks completed in this iteration
        progress_file = self.config.get_progress_file()
        recently_completed_ids = set()
        
        if Path(progress_file).exists():
            with open(progress_file, 'r') as f:
                for line in f:
                    if f"ProcessChunksTask_Iteration_{self.iteration}" in line and "COMPLETED" in line:
                        parts = line.strip().split(' | ')
                        if len(parts) >= 2:
                            hierarchical_id = parts[1]
                            if hierarchical_id != "GLOBAL":
                                recently_completed_ids.add(hierarchical_id)
        
        # Get these chunks from database
        all_completed = db.get_chunks_by_status(ChunkStatus.COMPLETED)
        recently_completed = [c for c in all_completed if c.hierarchical_id in recently_completed_ids]
        recently_completed.sort(key=lambda x: x.hierarchical_id)
        
        return recently_completed
    
    def _check_batch_quality(self, llm_client: LLMClient, batch: List[ChunkData], toc_summary: str) -> List[dict]:
        """Check quality of a batch of chunks"""
        # Prepare batch content for analysis
        batch_content = self._format_batch_for_analysis(batch)
        
        # Create quality check prompt
        prompt = self._create_quality_prompt(batch_content, toc_summary)
        
        # Get LLM analysis
        analysis = llm_client.complete(prompt)
        
        # Parse issues from analysis
        issues = self._parse_quality_issues(analysis, batch)
        
        return issues
    
    def _format_batch_for_analysis(self, batch: List[ChunkData]) -> str:
        """Format batch chunks for quality analysis"""
        formatted_chunks = []
        
        for chunk in batch:
            chunk_text = f"""
CHUNK {chunk.hierarchical_id}: {chunk.title}
STRESZCZENIE: {chunk.summary}
TREŚĆ: {chunk.content}
---"""
            formatted_chunks.append(chunk_text)
        
        return "\n".join(formatted_chunks)
    
    def _create_quality_prompt(self, batch_content: str, toc_summary: str) -> str:
        """Create quality check prompt"""
        prompt_config = self.task_config['prompt']
        
        user_prompt = prompt_config['user'].format(
            toc_summary=toc_summary,
            batch_content=batch_content
        )
        
        return f"{prompt_config['system']}\n\n{user_prompt}"
    
    def _parse_quality_issues(self, analysis: str, batch: List[ChunkData]) -> List[dict]:
        """Parse quality issues from LLM analysis"""
        issues = []
        
        # Simple parsing - look for hierarchical IDs mentioned as problematic
        lines = analysis.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for hierarchical IDs in the analysis
            for chunk in batch:
                if chunk.hierarchical_id in line and any(keyword in line.lower() for keyword in ['problem', 'błąd', 'niespójn', 'przepis', 'poprawi']):
                    current_issue = {
                        'hierarchical_id': chunk.hierarchical_id,
                        'chunk_id': chunk.id,
                        'issue_description': line,
                        'action': 'rewrite' if 'przepis' in line.lower() else 'review'
                    }
                    issues.append(current_issue)
        
        return issues
    
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
                f.write(f"  DESCRIPTION: {issue['issue_description']}\n")
    
    def _mark_chunks_for_revision(self, db: DatabaseManager, issues: List[dict]):
        """Mark chunks with issues for revision"""
        for issue in issues:
            chunk_id = issue['chunk_id']
            hierarchical_id = issue['hierarchical_id']
            action = issue['action']
            
            self._persist_task_progress(hierarchical_id, "QualityCheckTask", f"NEEDS_{action.upper()}")
    
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