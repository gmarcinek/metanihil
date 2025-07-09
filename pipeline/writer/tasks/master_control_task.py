import luigi
from pathlib import Path
from typing import List

from pipeline.writer.tasks.base_writer_task import BaseWriterTask
from writer.writer_service import WriterService
from writer.models import ChunkStatus
from pipeline.writer.config_loader import ConfigLoader


class MasterControlTask(BaseWriterTask):
    toc_path = luigi.Parameter()
    batch_size = luigi.IntParameter(default=5)
    max_iterations = luigi.IntParameter(default=550)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/master_control"
        self.iteration = 0
    
    @property 
    def task_name(self) -> str:
        return "master_control"

    def output(self):
        return luigi.LocalTarget(f"{self.output_dir}/master_completed.flag")
    
    def run(self):
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Persist task start
        self._persist_task_progress("GLOBAL", "MasterControlTask", "STARTED")
        
        try:
            # Initialize WriterService with consistent storage_dir
            writer_service = WriterService(storage_dir="output/writer_storage")
            
            # First, ensure TOC is parsed
            self._ensure_toc_parsed()
            
            # Reload WriterService after TOC setup to see fresh data
            writer_service = WriterService(storage_dir="output/writer_storage")
            
            # Main processing loop
            while self.iteration < self.max_iterations:
                self.iteration += 1
                
                print(f"\n🔄 ITERATION {self.iteration}")
                self._persist_task_progress("GLOBAL", f"MasterControlTask_Iteration_{self.iteration}", "STARTED")
                
                # Check what needs to be done
                status = self._assess_current_status(writer_service)
                
                if status == "all_completed":
                    print("✅ All chunks processed successfully!")
                    self._run_final_qa()
                    break
                    
                elif status == "needs_processing":
                    print(f"📝 Processing next batch of {self.batch_size} chunks...")
                    self._run_processing_batch()
                    # Reload WriterService to see updates
                    writer_service = WriterService(storage_dir="output/writer_storage")
                    
                elif status == "needs_revision":
                    print("🔧 Running revision for problematic chunks...")
                    self._run_revision_batch()
                    # Reload WriterService to see updates
                    writer_service = WriterService(storage_dir="output/writer_storage")
                    
                elif status == "no_chunks":
                    print("❌ No chunks found in database!")
                    break
                
                self._persist_task_progress("GLOBAL", f"MasterControlTask_Iteration_{self.iteration}", "COMPLETED")
                
                # Log current progress
                self._log_progress(writer_service)
            
            if self.iteration >= self.max_iterations:
                print(f"⚠️ Reached maximum iterations ({self.max_iterations})")
            
            # Create completion flag
            with open(self.output().path, 'w', encoding='utf-8') as f:
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
        
        # Run initial setup tasks with CLI parameters
        luigi.build([
            ParseTOCTask(
                toc_path=self.toc_path,
                author=self.author,
                title=self.title,
                custom_prompt=self.custom_prompt
            ),
            EmbedTOCTask(
                toc_path=self.toc_path,
                author=self.author,
                title=self.title,
                custom_prompt=self.custom_prompt
            ),
            CreateSummaryTask(
                toc_path=self.toc_path,
                author=self.author,
                title=self.title,
                custom_prompt=self.custom_prompt
            )
        ], local_scheduler=True)
        
        print("✅ TOC setup completed")
    
    def _assess_current_status(self, writer_service: WriterService) -> str:
        """Assess what needs to be done next"""
        print(f"🔍 DEBUG: Assessing status...")
        print(f"🔍 DEBUG: Total chunks in WriterService: {len(writer_service.chunks)}")
        
        # Debug: Print all chunk statuses
        status_debug = {}
        for chunk in writer_service.chunks.values():
            status_str = chunk.status.value
            status_debug[status_str] = status_debug.get(status_str, 0) + 1
        
        print(f"🔍 DEBUG: Chunk status distribution:")
        for status_str, count in status_debug.items():
            print(f"   {status_str}: {count}")
        
        # Check for chunks needing revision
        chunks_needing_revision = self._get_chunks_needing_revision()
        print(f"🔍 DEBUG: Chunks needing revision: {len(chunks_needing_revision)}")
        if chunks_needing_revision:
            print(f"🔍 DEBUG: Revision list: {chunks_needing_revision[:5]}...")  # Show first 5
            return "needs_revision"
        
        # Check for unprocessed chunks - using enum comparison
        not_started_chunks = writer_service.get_chunks_by_status(ChunkStatus.NOT_STARTED)
        print(f"🔍 DEBUG: NOT_STARTED chunks found: {len(not_started_chunks)}")
        if not_started_chunks:
            print(f"🔍 DEBUG: First few NOT_STARTED chunks:")
            for chunk in not_started_chunks[:3]:
                print(f"   {chunk.hierarchical_id}: {chunk.title} (status: {chunk.status.value})")
            return "needs_processing"
        
        # Check if we have any chunks at all
        all_chunks = list(writer_service.chunks.values())
        print(f"🔍 DEBUG: Total chunks found: {len(all_chunks)}")
        if not all_chunks:
            return "no_chunks"
        
        # Check completed chunks
        completed_chunks = writer_service.get_chunks_by_status(ChunkStatus.COMPLETED)
        print(f"🔍 DEBUG: COMPLETED chunks: {len(completed_chunks)}")
        
        # All chunks are completed
        print(f"🔍 DEBUG: All chunks completed - ready for final QA")
        return "all_completed"
    
    def _run_processing_batch(self):
        """Run processing for next batch of chunks"""
        from .process_chunks_task import ProcessChunksTask
        from .quality_check_task import QualityCheckTask
        
        # Create dynamic task instances for this iteration with CLI parameters
        process_task = ProcessChunksTask(
            toc_path=self.toc_path,
            batch_size=self.batch_size,
            iteration=self.iteration,
            author=self.author,
            title=self.title,
            custom_prompt=self.custom_prompt
        )
        
        quality_task = QualityCheckTask(
            toc_path=self.toc_path,
            iteration=self.iteration,
            author=self.author,
            title=self.title,
            custom_prompt=self.custom_prompt
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
            iteration=self.iteration,
            author=self.author,
            title=self.title,
            custom_prompt=self.custom_prompt
        )
        
        result = luigi.build([revision_task], local_scheduler=True)
        
        if not result:
            raise Exception(f"Revision batch failed in iteration {self.iteration}")
    
    def _run_final_qa(self):
        """Run final QA when all chunks are completed"""
        from .final_qa_task import FinalQATask
        
        print("🎯 Running final QA analysis...")
        
        final_qa_task = FinalQATask(
            toc_path=self.toc_path,
            author=self.author,
            title=self.title,
            custom_prompt=self.custom_prompt
        )
        
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
        
        with open(progress_file, 'r', encoding='utf-8') as f:
            for line in f:
                if 'QualityCheckTask' in line and ('NEEDS_REWRITE' in line or 'NEEDS_REVIEW' in line):
                    parts = line.strip().split(' | ')
                    if len(parts) >= 2:
                        hierarchical_id = parts[1]
                        if hierarchical_id not in chunks_to_revise and hierarchical_id != "GLOBAL":
                            chunks_to_revise.append(hierarchical_id)
        
        return chunks_to_revise
    
    def _log_progress(self, writer_service: WriterService):
        """Log current progress"""
        not_started = len(writer_service.get_chunks_by_status(ChunkStatus.NOT_STARTED))
        in_progress = len(writer_service.get_chunks_by_status(ChunkStatus.IN_PROGRESS))
        completed = len(writer_service.get_chunks_by_status(ChunkStatus.COMPLETED))
        failed = len(writer_service.get_chunks_by_status(ChunkStatus.FAILED))
        
        total = not_started + in_progress + completed + failed
        
        if total > 0:
            completion_rate = (completed / total) * 100
            print(f"📊 Progress: {completed}/{total} completed ({completion_rate:.1f}%)")
            print(f"   Not started: {not_started}, In progress: {in_progress}, Failed: {failed}")
        
        # Save progress to file
        progress_log = f"{self.output_dir}/progress_log.txt"
        with open(progress_log, 'a', encoding='utf-8') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | Iteration {self.iteration} | Completed: {completed}/{total} ({completion_rate:.1f}%)\n")
    
    def _persist_task_progress(self, hierarchical_id: str, task_name: str, status: str):
        """Persist task progress to file"""
        progress_file = self.config.get_progress_file()
        Path(progress_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(progress_file, 'a', encoding='utf-8') as f:
            from datetime import datetime
            timestamp = datetime.now().isoformat()
            f.write(f"{timestamp} | {hierarchical_id} | {task_name} | {status}\n")