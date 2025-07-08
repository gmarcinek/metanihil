#!/usr/bin/env python3
"""
Writer Pipeline Runner
Usage: python -m pipeline.writer.run_pipeline <toc_file_path> [options]
"""

import sys
import argparse
import luigi
from pathlib import Path

from pipeline.writer.tasks.master_control_task import MasterControlTask


def main():
    parser = argparse.ArgumentParser(description='Run Writer Pipeline')
    parser.add_argument('toc_path', help='Path to TOC file')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of chunks to process per batch (default: 5)')
    parser.add_argument('--max-iterations', type=int, default=100, help='Maximum number of iterations (default: 100)')
    parser.add_argument('--local-scheduler', action='store_true', default=True, help='Use Luigi local scheduler')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes (default: 1)')
    
    args = parser.parse_args()
    
    # Validate TOC file exists
    toc_file = Path(args.toc_path)
    if not toc_file.exists():
        print(f"❌ TOC file not found: {args.toc_path}")
        sys.exit(1)
    
    print(f"🚀 Starting Writer Pipeline")
    print(f"   TOC file: {args.toc_path}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Max iterations: {args.max_iterations}")
    print(f"   Workers: {args.workers}")
    print()
    
    # Create master control task
    master_task = MasterControlTask(
        toc_path=args.toc_path,
        batch_size=args.batch_size,
        max_iterations=args.max_iterations
    )
    
    # Run pipeline
    try:
        result = luigi.build(
            [master_task],
            local_scheduler=args.local_scheduler,
            workers=args.workers,
            detailed_summary=True
        )
        
        if result:
            print("\n✅ Pipeline completed successfully!")
            _print_completion_summary(args.toc_path)
        else:
            print("\n❌ Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        _print_resume_instructions(args.toc_path, args.batch_size, args.max_iterations)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        _print_resume_instructions(args.toc_path, args.batch_size, args.max_iterations)
        sys.exit(1)


def _print_completion_summary(toc_path: str):
    """Print completion summary with output locations"""
    toc_name = Path(toc_path).stem
    
    print("📁 Output locations:")
    print(f"   Main output: output/{toc_name}/")
    print(f"   TOC summary: output/{toc_name}/toc_short.txt")
    print(f"   Progress log: output/task_progress.txt")
    print(f"   Database: output/writer.db")
    
    # Check if final QA was run
    final_qa_path = Path(f"output/{toc_name}/final_qa/final_qa_completed.flag")
    if final_qa_path.exists():
        print(f"   Final QA report: output/{toc_name}/final_qa/final_qa_report.txt")
        print(f"   Change map: output/{toc_name}/final_qa/hierarchical_change_map.txt")


def _print_resume_instructions(toc_path: str, batch_size: int, max_iterations: int):
    """Print instructions for resuming pipeline"""
    print("\n🔄 To resume pipeline from where it left off:")
    print(f"   python -m pipeline.writer.run_pipeline {toc_path} --batch-size {batch_size} --max-iterations {max_iterations}")
    print("\n📊 To check current progress:")
    print("   tail -f output/task_progress.txt")
    print("\n📂 To check outputs so far:")
    toc_name = Path(toc_path).stem
    print(f"   ls -la output/{toc_name}/")


def run_single_task():
    """Helper function to run individual tasks for debugging"""
    parser = argparse.ArgumentParser(description='Run individual Writer Pipeline task')
    parser.add_argument('task', choices=['parse', 'embed', 'summary', 'process', 'quality', 'revision', 'final-qa'], 
                       help='Task to run')
    parser.add_argument('toc_path', help='Path to TOC file')
    parser.add_argument('--iteration', type=int, default=1, help='Iteration number (for process/quality/revision)')
    parser.add_argument('--batch-size', type=int, default=5, help='Batch size (for process task)')
    
    args = parser.parse_args()
    
    if args.task == 'parse':
        from pipeline.writer.tasks.parse_toc_task import ParseTOCTask
        task = ParseTOCTask(toc_path=args.toc_path)
    elif args.task == 'embed':
        from pipeline.writer.tasks.embed_toc_task import EmbedTOCTask
        task = EmbedTOCTask(toc_path=args.toc_path)
    elif args.task == 'summary':
        from pipeline.writer.tasks.create_summary_task import CreateSummaryTask
        task = CreateSummaryTask(toc_path=args.toc_path)
    elif args.task == 'process':
        from pipeline.writer.tasks.process_chunks_task import ProcessChunksTask
        task = ProcessChunksTask(toc_path=args.toc_path, iteration=args.iteration, batch_size=args.batch_size)
    elif args.task == 'quality':
        from pipeline.writer.tasks.quality_check_task import QualityCheckTask
        task = QualityCheckTask(toc_path=args.toc_path, iteration=args.iteration)
    elif args.task == 'revision':
        from pipeline.writer.tasks.revision_task import RevisionTask
        task = RevisionTask(toc_path=args.toc_path, iteration=args.iteration)
    elif args.task == 'final-qa':
        from pipeline.writer.tasks.final_qa_task import FinalQATask
        task = FinalQATask(toc_path=args.toc_path)
    
    result = luigi.build([task], local_scheduler=True, detailed_summary=True)
    
    if result:
        print(f"✅ Task {args.task} completed successfully!")
    else:
        print(f"❌ Task {args.task} failed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'single':
        # Remove 'single' from args and run single task
        sys.argv.pop(1)
        run_single_task()
    else:
        main()