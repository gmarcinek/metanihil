import luigi
from pathlib import Path
from typing import List

from components.structured_task import StructuredTask
from pipeline.writer.database import DatabaseManager
from pipeline.writer.models import ChunkStatus, ChunkData
from pipeline.writer.config_loader import ConfigLoader
from llm import LLMClient


class FinalQATask(StructuredTask):
    toc_path = luigi.Parameter()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.task_config = self.config.get_task_config('FinalQATask')
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/final_qa"
    
    def requires(self):
        # This should be called when all chunks are processed
        # Logic to determine this will be in the master control
        pass
    
    def output(self):
        return luigi.LocalTarget(f"{self.output_dir}/final_qa_completed.flag")
    
    def run(self):
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Persist task start
        self._persist_task_progress("GLOBAL", "FinalQATask", "STARTED")
        
        try:
            # Initialize components
            db = DatabaseManager(self.config.get_database_path())
            llm_client = LLMClient(model=self.task_config['model'])  # gpt-4.1-mini
            
            # Get ALL completed chunks
            all_chunks = db.get_chunks_by_status(ChunkStatus.COMPLETED)
            all_chunks.sort(key=lambda x: x.hierarchical_id)
            
            if not all_chunks:
                raise ValueError("No completed chunks found for final QA")
            
            # Load TOC summary
            toc_summary = self._load_toc_summary()
            
            # Prepare full document for analysis
            full_document = self._prepare_full_document(all_chunks)
            
            # Run final QA analysis
            qa_analysis = self._run_final_qa_analysis(llm_client, full_document, toc_summary, all_chunks)
            
            # Parse inconsistencies and improvement suggestions
            issues = self._parse_final_qa_issues(qa_analysis, all_chunks)
            
            # Create comprehensive report
            self._create_final_qa_report(all_chunks, qa_analysis, issues)
            
            # Create hierarchical change suggestions
            self._create_hierarchical_change_map(issues)
            
            # Create completion flag
            with self.output().open('w') as f:
                f.write(f"Final QA completed for {len(all_chunks)} chunks. Found {len(issues)} issues.")
            
            # Persist task completion
            self._persist_task_progress("GLOBAL", "FinalQATask", "COMPLETED")
            
            print(f"✅ Final QA completed. Found {len(issues)} issues for review.")
            
        except Exception as e:
            self._persist_task_progress("GLOBAL", "FinalQATask", "FAILED")
            raise
    
    def _prepare_full_document(self, chunks: List[ChunkData]) -> str:
        """Prepare full document text for analysis"""
        document_parts = []
        
        for chunk in chunks:
            chunk_text = f"\n--- {chunk.hierarchical_id}: {chunk.title} ---\n{chunk.content}"
            document_parts.append(chunk_text)
        
        return "\n".join(document_parts)
    
    def _run_final_qa_analysis(self, llm_client: LLMClient, full_document: str, toc_summary: str, chunks: List[ChunkData]) -> str:
        """Run comprehensive QA analysis on entire document"""
        prompt = self._create_final_qa_prompt(full_document, toc_summary, len(chunks))
        
        return llm_client.complete(prompt)
    
    def _create_final_qa_prompt(self, full_document: str, toc_summary: str, chunk_count: int) -> str:
        """Create final QA prompt"""
        prompt_config = self.task_config['prompt']
        
        user_prompt = prompt_config['user'].format(
            toc_summary=toc_summary,
            chunk_count=chunk_count,
            full_document=full_document
        )
        
        return f"{prompt_config['system']}\n\n{user_prompt}"
    
    def _parse_final_qa_issues(self, qa_analysis: str, chunks: List[ChunkData]) -> List[dict]:
        """Parse issues from final QA analysis"""
        issues = []
        
        # Look for hierarchical IDs mentioned in analysis
        chunk_ids = {chunk.hierarchical_id for chunk in chunks}
        
        lines = qa_analysis.split('\n')
        current_issue = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for hierarchical IDs and issue keywords
            for chunk_id in chunk_ids:
                if chunk_id in line and any(keyword in line.lower() for keyword in 
                    ['niespójn', 'problem', 'błąd', 'poprawi', 'zmieni', 'dodaj', 'usuń']):
                    
                    issue_type = self._classify_issue_type(line)
                    
                    issue = {
                        'hierarchical_id': chunk_id,
                        'issue_type': issue_type,
                        'description': line,
                        'severity': self._assess_severity(line),
                        'suggested_action': self._extract_suggested_action(line)
                    }
                    issues.append(issue)
        
        return issues
    
    def _classify_issue_type(self, line: str) -> str:
        """Classify type of issue"""
        line_lower = line.lower()
        
        if any(word in line_lower for word in ['niespójn', 'sprzeczn']):
            return 'inconsistency'
        elif any(word in line_lower for word in ['brak', 'dodaj']):
            return 'missing_content'
        elif any(word in line_lower for word in ['zbędny', 'usuń']):
            return 'redundant_content'
        elif any(word in line_lower for word in ['przepływ', 'kolejność']):
            return 'flow_issue'
        else:
            return 'general_improvement'
    
    def _assess_severity(self, line: str) -> str:
        """Assess issue severity"""
        line_lower = line.lower()
        
        if any(word in line_lower for word in ['krytyczny', 'poważny', 'błąd']):
            return 'high'
        elif any(word in line_lower for word in ['drobny', 'minor', 'lekki']):
            return 'low'
        else:
            return 'medium'
    
    def _extract_suggested_action(self, line: str) -> str:
        """Extract suggested action from line"""
        line_lower = line.lower()
        
        if 'przepis' in line_lower:
            return 'rewrite'
        elif 'dodaj' in line_lower:
            return 'add_content'
        elif 'usuń' in line_lower:
            return 'remove_content'
        elif 'zmień' in line_lower or 'poprawi' in line_lower:
            return 'modify'
        else:
            return 'review'
    
    def _create_final_qa_report(self, chunks: List[ChunkData], qa_analysis: str, issues: List[dict]):
        """Create comprehensive final QA report"""
        report_file = f"{self.output_dir}/final_qa_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("FINAL QUALITY ASSURANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Document statistics
            f.write("DOCUMENT STATISTICS:\n")
            f.write(f"- Total chunks: {len(chunks)}\n")
            f.write(f"- Total issues found: {len(issues)}\n")
            
            # Issue breakdown by type
            issue_types = {}
            severity_counts = {}
            
            for issue in issues:
                issue_type = issue['issue_type']
                severity = issue['severity']
                
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            f.write(f"\nISSUE BREAKDOWN BY TYPE:\n")
            for issue_type, count in issue_types.items():
                f.write(f"- {issue_type}: {count}\n")
            
            f.write(f"\nISSUE BREAKDOWN BY SEVERITY:\n")
            for severity, count in severity_counts.items():
                f.write(f"- {severity}: {count}\n")
            
            # Full LLM analysis
            f.write(f"\n\nFULL QA ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(qa_analysis)
            
            # Detailed issues list
            f.write(f"\n\nDETAILED ISSUES LIST:\n")
            f.write("-" * 30 + "\n")
            
            for i, issue in enumerate(issues, 1):
                f.write(f"\n{i}. CHUNK {issue['hierarchical_id']}\n")
                f.write(f"   Type: {issue['issue_type']}\n")
                f.write(f"   Severity: {issue['severity']}\n")
                f.write(f"   Action: {issue['suggested_action']}\n")
                f.write(f"   Description: {issue['description']}\n")
    
    def _create_hierarchical_change_map(self, issues: List[dict]):
        """Create hierarchical map of suggested changes"""
        change_map_file = f"{self.output_dir}/hierarchical_change_map.txt"
        
        # Group issues by hierarchical_id
        issues_by_chunk = {}
        for issue in issues:
            chunk_id = issue['hierarchical_id']
            if chunk_id not in issues_by_chunk:
                issues_by_chunk[chunk_id] = []
            issues_by_chunk[chunk_id].append(issue)
        
        with open(change_map_file, 'w', encoding='utf-8') as f:
            f.write("HIERARCHICAL CHANGE MAP\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("FORMAT: chunk_id | issue_type | severity | action | description\n")
            f.write("-" * 80 + "\n\n")
            
            # Sort by hierarchical ID
            for chunk_id in sorted(issues_by_chunk.keys()):
                chunk_issues = issues_by_chunk[chunk_id]
                
                f.write(f"CHUNK {chunk_id}:\n")
                for issue in chunk_issues:
                    f.write(f"  • {issue['issue_type']} | {issue['severity']} | {issue['suggested_action']}\n")
                    f.write(f"    {issue['description']}\n")
                f.write("\n")
            
            if not issues_by_chunk:
                f.write("No issues found. Document quality is satisfactory.\n")
    
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