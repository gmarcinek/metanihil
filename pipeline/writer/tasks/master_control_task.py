import luigi
from pathlib import Path
from typing import List

from components.structured_task import StructuredTask
from pipeline.writer.database import DatabaseManager
from pipeline.writer.models import ChunkStatus
from pipeline.writer.config_loader import ConfigLoader


class MasterControlTask(StructuredTask):
    toc_path = luigi.Parameter()
    batch_size = luigi.IntParameter(default=5)
    max_iterations = luigi.IntParameter(default=100)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/master_control"
        self.iteration = 0
    
    def output(self):
        return luigi.LocalTarget(f"{self.output_dir}/master_completed.flag")
    
    def run(self):
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Persist task start
        self._persist_task_progress("GLOBAL", "MasterControlTask", "STARTED")
        
        try:
            # Initialize database
            db = DatabaseManager(self.config.get_database_path())
            
            # First, ensure TOC is parsed
            self._ensure_toc_parsed()
            
            # Main processing loop
            while self.iteration < self.max_iterations:
                self.iteration += 1
                
                print(f"\n🔄 ITERATION {self.iteration}")
                self._persist_task_progress("GLOBAL", f"MasterControlTask_Iteration_{self.iteration}", "STARTED")
                
                # Check what needs to be done
                status = self._assess_current_status(db)
                
                if status == "all_completed":
                    print("✅ All chunks processed successfully!")
                    self._run_final_qa()
                    break
                    
                elif status == "needs_processing":
                    print(f"📝 Processing next batch of {self.batch_size} chunks...")
                    self._run_processing_batch()
                    
                elif status == "needs_revision":
                    print("🔧 Running revision for problematic chunks...")
                    self._run_revision_batch()
                    
                elif status == "no_chunks":
                    print("❌ No chunks found in database!")
                    break
                
                self._persist_task_progress("GLOBAL", f"MasterControlTask_Iteration_{self.iteration}", "COMPLETED")
                
                # Log current progress
                self._log_progress(db)
            
            if self.iteration >= self.max_iterations:
                print(f"⚠️ Reached maximum iterations ({self.max_iterations})")
            
            # Create completion flag
            with self.output().open('w') as f:
                f.write(f"Master control completed after {self.iteration} iterations")
            
            # Persist task completion
            self._persist_task_progress("GLOBAL", "MasterControlTask", "COMPLETED")
            
        except Exception as e:
            self._persist_task_progress("GLOBAL", "MasterControlTask", "FAILED")
            raise
    
    def _ensure_toc_parsed(self):
        """Ensure TOC is parsed before starting processing"""
        from .parse_toc_task import ParseTOCTask
        from .embed_toc_task import EmbedTOCTask
        from .create_summary_task import CreateSummaryTask
        
        print("🔍 Ensuring TOC is parsed and embedded...")
        
        # Run initial setup tasks
        luigi.build([
            ParseTOCTask(toc_path=self.toc_path),
            EmbedTOCTask(toc_path=self.toc_path),
            CreateSummaryTask(toc_path=self.toc_path)
        ], local_scheduler=True)
        
        print("✅ TOC setup completed")
    
    def _assess_current_status(self, db: DatabaseManager) -> str:
        """Assess what needs to be done next"""
        # Check for chunks needing revision
        chunks_needing_revision = self._get_chunks_needing_revision()
        if chunks_needing_revision:
            return "needs_revision"
        
        # Check for unprocessed chunks
        not_started_chunks = db.get_chunks_by_status(ChunkStatus.NOT_STARTED)
        if not_started_chunks:
            return "needs_processing"
        
        # Check if we have any chunks at all
        completed_chunks = db.get_chunks_by_status(ChunkStatus.COMPLETED)
        if not completed_chunks:
            return "no_chunks"
        
        # All chunks are completed
        return "all_completed"
    
    def _run_processing_batch(self):
        """Run processing for next batch of chunks"""
        from .process_chunks_task import ProcessChunksTask
        from .quality_check_task import QualityCheckTask
        
        # Create dynamic task instances for this iteration
        process_task = ProcessChunksTask(
            toc_path=self.toc_path,
            batch_size=self.batch_size,
            iteration=self.iteration
        )
        
        quality_task = QualityCheckTask(
            toc_path=self.toc_path,
            iteration=self.iteration
        )
        
        # Run processing and quality check
        result = luigi.build([process_task, quality_task], local_scheduler=True)
        
        if not result:
            raise Exception(f"Processing batch failed in iteration {self.iteration}")
    
    def _run_revision_batch(self):
        """Run revision for problematic chunks"""
        from .revision_task import RevisionTask
        
        revision_task = RevisionTask(
            toc_path=self.toc_path,
            iteration=self.iteration
        )
        
        result = luigi.build([revision_task], local_scheduler=True)
        
        if not result:
            raise Exception(f"Revision batch failed in iteration {self.iteration}")
    
    def _run_final_qa(self):
        """Run final QA when all chunks are completed"""
        from .final_qa_task import FinalQATask
        
        print("🎯 Running final QA analysis...")
        
        final_qa_task = FinalQATask(toc_path=self.toc_path)
        
        result = luigi.build([final_qa_task], local_scheduler=True)
        
        if result:
            print("✅ Final QA completed!")
        else:
            print("❌ Final QA failed!")
    
    def _get_chunks_needing_revision(self) -> List[str]:
        """Get chunks that need revision from progress file"""
        chunks_to_revise = []
        progress_file = self.config.get_progress_file()
        
        if not Path(progress_file).exists():
            return chunks_to_revise
        
        with open(progress_file, 'r') as f:
            for line in f:
                if 'QualityCheckTask' in line and ('NEEDS_REWRITE' in line or 'NEEDS_REVIEW' in line):
                    parts = line.strip().split(' | ')
                    if len(parts) >= 2:
                        hierarchical_id = parts[1]
                        if hierarchical_id not in chunks_to_revise and hierarchical_id != "GLOBAL":
                            chunks_to_revise.append(hierarchical_id)
        
        return chunks_to_revise
    
    def _log_progress(self, db: DatabaseManager):
        """Log current progress"""
        not_started = len(db.get_chunks_by_status(ChunkStatus.NOT_STARTED))
        in_progress = len(db.get_chunks_by_status(ChunkStatus.IN_PROGRESS))
        completed = len(db.get_chunks_by_status(ChunkStatus.COMPLETED))
        failed = len(db.get_chunks_by_status(ChunkStatus.FAILED))
        
        total = not_started + in_progress + completed + failed
        
        if total > 0:
            completion_rate = (completed / total) * 100
            print(f"📊 Progress: {completed}/{total} completed ({completion_rate:.1f}%)")
            print(f"   Not started: {not_started}, In progress: {in_progress}, Failed: {failed}")
        
        # Save progress to file
        progress_log = f"{self.output_dir}/progress_log.txt"
        with open(progress_log, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | Iteration {self.iteration} | Completed: {completed}/{total} ({completion_rate:.1f}%)\n")
    
    def _persist_task_progress(self, hierarchical_id: str, task_name: str, status: str):
        """Persist task progress to file"""
        progress_file = self.config.get_progress_file()
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'a') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | {hierarchical_id} | {task_name} | {status}\n")