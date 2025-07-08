import luigi
from pathlib import Path
from typing import List

from pipeline.writer.tasks.base_writer_task import BaseWriterTask
from writer.writer_service import WriterService
from writer.models import ChunkStatus, ChunkData
from pipeline.writer.config_loader import ConfigLoader
from llm import LLMClient


class FinalQATask(BaseWriterTask):
    toc_path = luigi.Parameter()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.toc_name = Path(self.toc_path).stem
        self.config = ConfigLoader()
        self.task_config = self.config.get_task_config('FinalQATask')
        self.output_dir = f"{self.config.get_output_config()['base_dir']}/{self.toc_name}/final_qa"
    
    @property 
    def task_name(self) -> str:
        return "final_qa"
    
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
            # Initialize WriterService and LLM
            writer_service = WriterService(storage_dir="output/writer_storage")
            llm_client = LLMClient(model=self.task_config['model'])
            
            # Get ALL completed chunks
            all_chunks = writer_service.get_chunks_by_status(ChunkStatus.COMPLETED)
            all_chunks.sort(key=lambda x: x.hierarchical_id)
            
            if not all_chunks:
                raise ValueError("No completed chunks found for final QA")
            
            # Load TOC summary
            toc_summary = self._load_toc_summary()
            
            # Prepare document segments for analysis (avoid token limits)
            document_segments = self._prepare_document_segments(all_chunks)
            
            # Run comprehensive QA analysis
            qa_results = self._run_comprehensive_qa_analysis(
                llm_client, writer_service, document_segments, toc_summary, all_chunks
            )
            
            # Parse and categorize issues
            all_issues = self._parse_and_categorize_issues(qa_results, all_chunks)
            
            # Run semantic consistency analysis
            semantic_issues = self._analyze_semantic_consistency(writer_service, all_chunks)
            all_issues.extend(semantic_issues)
            
            # Create comprehensive reports
            self._create_final_qa_report(all_chunks, qa_results, all_issues)
            self._create_hierarchical_change_map(all_issues)
            self._create_semantic_analysis_report(writer_service, all_chunks)
            
            # Create completion flag
            with open(self.output().path, 'w', encoding='utf-8') as f:
                f.write(f"Final QA completed for {len(all_chunks)} chunks. Found {len(all_issues)} issues.")
            
            # Persist task completion
            self._persist_task_progress("GLOBAL", "FinalQATask", "COMPLETED")
            
            print(f"✅ Final QA completed. Found {len(all_issues)} issues for review.")
            
        except Exception as e:
            self._persist_task_progress("GLOBAL", "FinalQATask", "FAILED")
            raise
    
    def _prepare_document_segments(self, chunks: List[ChunkData]) -> List[dict]:
        """Prepare document segments to avoid token limits"""
        segments = []
        current_segment = []
        current_length = 0
        max_segment_length = 15000  # chars per segment to stay under token limits
        
        for chunk in chunks:
            chunk_text = f"\n--- {chunk.hierarchical_id}: {chunk.title} ---\n{chunk.content or '[No content]'}"
            chunk_length = len(chunk_text)
            
            if current_length + chunk_length > max_segment_length and current_segment:
                # Finalize current segment
                segments.append({
                    'content': '\n'.join(current_segment),
                    'chunk_ids': [c.hierarchical_id for c in chunks[len(segments)*10:(len(segments)+1)*10]]
                })
                current_segment = [chunk_text]
                current_length = chunk_length
            else:
                current_segment.append(chunk_text)
                current_length += chunk_length
        
        # Add final segment
        if current_segment:
            segments.append({
                'content': '\n'.join(current_segment),
                'chunk_ids': [chunk.hierarchical_id for chunk in chunks if chunk.content]
            })
        
        print(f"📄 Document split into {len(segments)} segments for analysis")
        return segments
    
    def _run_comprehensive_qa_analysis(self, llm_client: LLMClient, writer_service: WriterService,
                                     document_segments: List[dict], toc_summary: str, 
                                     all_chunks: List[ChunkData]) -> List[str]:
        """Run QA analysis on each document segment"""
        qa_results = []
        
        for i, segment in enumerate(document_segments, 1):
            print(f"🔍 Analyzing document segment {i}/{len(document_segments)}")
            
            # Create segment-specific prompt
            prompt = self._create_final_qa_prompt(segment['content'], toc_summary, len(all_chunks), i, len(document_segments))
            
            # Get LLM analysis
            analysis = llm_client.chat(prompt)
            qa_results.append(analysis)
        
        return qa_results
    
    def _create_final_qa_prompt(self, segment_content: str, toc_summary: str, 
                              total_chunks: int, segment_num: int, total_segments: int) -> str:
        """Create final QA prompt for document segment"""
        prompt_config = self.task_config['prompt']
        
        user_prompt = prompt_config['user'].format(
            toc_summary=toc_summary,
            chunk_count=total_chunks,
            full_document=segment_content
        )
        
        # Add segment context
        segment_context = f"\n\nSEGMENT KONTEKST: Analizujesz segment {segment_num}/{total_segments} dokumentu."
        enhanced_prompt = user_prompt + segment_context
        
        return f"{prompt_config['system']}\n\n{enhanced_prompt}"
    
    def _analyze_semantic_consistency(self, writer_service: WriterService, all_chunks: List[ChunkData]) -> List[dict]:
        """Analyze semantic consistency using WriterService search capabilities"""
        semantic_issues = []
        
        print("🔍 Analyzing semantic consistency...")
        
        # Check for semantic inconsistencies
        for chunk in all_chunks:
            if not chunk.content:
                continue
            
            # Find semantically similar chunks
            similar_results = writer_service.search_chunks_semantic(
                query=chunk.content[:400],
                max_results=5,
                min_similarity=0.8  # High similarity threshold
            )
            
            # Check for potential inconsistencies
            for result in similar_results:
                if result.chunk.id != chunk.id and result.similarity_score > 0.85:
                    # Very similar content - potential duplication or inconsistency
                    issue = {
                        'type': 'semantic_similarity',
                        'hierarchical_id': chunk.hierarchical_id,
                        'related_chunk': result.chunk.hierarchical_id,
                        'similarity_score': result.similarity_score,
                        'issue_description': f"Very high similarity ({result.similarity_score:.2f}) with {result.chunk.hierarchical_id}",
                        'action': 'review',
                        'severity': 'medium'
                    }
                    semantic_issues.append(issue)
        
        print(f"🔍 Found {len(semantic_issues)} semantic consistency issues")
        return semantic_issues
    
    def _parse_and_categorize_issues(self, qa_results: List[str], all_chunks: List[ChunkData]) -> List[dict]:
        """Parse issues from QA analysis results"""
        all_issues = []
        
        # Create lookup for chunks
        chunk_lookup = {chunk.hierarchical_id: chunk for chunk in all_chunks}
        
        for segment_analysis in qa_results:
            lines = segment_analysis.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for hierarchical IDs mentioned in analysis
                for hierarchical_id in chunk_lookup.keys():
                    if hierarchical_id in line and any(keyword in line.lower() for keyword in 
                        ['niespójn', 'problem', 'błąd', 'poprawi', 'zmieni', 'dodaj', 'usuń']):
                        
                        issue_type = self._classify_issue_type(line)
                        severity = self._assess_severity(line)
                        action = self._extract_suggested_action(line)
                        
                        issue = {
                            'type': issue_type,
                            'hierarchical_id': hierarchical_id,
                            'issue_description': line,
                            'severity': severity,
                            'action': action,
                            'source': 'llm_analysis'
                        }
                        all_issues.append(issue)
        
        return all_issues
    
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
    
    def _create_final_qa_report(self, chunks: List[ChunkData], qa_results: List[str], issues: List[dict]):
        """Create comprehensive final QA report"""
        report_file = f"{self.output_dir}/final_qa_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("FINAL QUALITY ASSURANCE REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Document statistics
            f.write("DOCUMENT STATISTICS:\n")
            f.write(f"- Total chunks: {len(chunks)}\n")
            f.write(f"- Total issues found: {len(issues)}\n")
            f.write(f"- QA analysis segments: {len(qa_results)}\n")
            
            # Issue breakdown by type
            issue_types = {}
            severity_counts = {}
            source_counts = {}
            
            for issue in issues:
                issue_type = issue.get('type', 'unknown')
                severity = issue.get('severity', 'unknown')
                source = issue.get('source', 'unknown')
                
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1
            
            f.write(f"\nISSUE BREAKDOWN BY TYPE:\n")
            for issue_type, count in sorted(issue_types.items()):
                f.write(f"- {issue_type}: {count}\n")
            
            f.write(f"\nISSUE BREAKDOWN BY SEVERITY:\n")
            for severity, count in sorted(severity_counts.items()):
                f.write(f"- {severity}: {count}\n")
            
            f.write(f"\nISSUE BREAKDOWN BY SOURCE:\n")
            for source, count in sorted(source_counts.items()):
                f.write(f"- {source}: {count}\n")
            
            # Full LLM analysis
            f.write(f"\n\nFULL QA ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            for i, analysis in enumerate(qa_results, 1):
                f.write(f"\nSEGMENT {i} ANALYSIS:\n")
                f.write(analysis)
                f.write(f"\n{'-'*20}\n")
            
            # Detailed issues list
            f.write(f"\n\nDETAILED ISSUES LIST:\n")
            f.write("-" * 30 + "\n")
            
            for i, issue in enumerate(issues, 1):
                f.write(f"\n{i}. CHUNK {issue['hierarchical_id']}\n")
                f.write(f"   Type: {issue.get('type', 'unknown')}\n")
                f.write(f"   Severity: {issue.get('severity', 'unknown')}\n")
                f.write(f"   Action: {issue.get('action', 'unknown')}\n")
                f.write(f"   Source: {issue.get('source', 'unknown')}\n")
                f.write(f"   Description: {issue['issue_description']}\n")
                
                if 'related_chunk' in issue:
                    f.write(f"   Related chunk: {issue['related_chunk']} (similarity: {issue.get('similarity_score', 0):.2f})\n")
    
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
                    f.write(f"  • {issue.get('type', 'unknown')} | {issue.get('severity', 'unknown')} | {issue.get('action', 'unknown')}\n")
                    f.write(f"    {issue['issue_description']}\n")
                    if 'related_chunk' in issue:
                        f.write(f"    Related: {issue['related_chunk']} (similarity: {issue.get('similarity_score', 0):.2f})\n")
                f.write("\n")
            
            if not issues_by_chunk:
                f.write("No issues found. Document quality is satisfactory.\n")
    
    def _create_semantic_analysis_report(self, writer_service: WriterService, chunks: List[ChunkData]):
        """Create semantic analysis report using WriterService capabilities"""
        semantic_report_file = f"{self.output_dir}/semantic_analysis_report.txt"
        
        with open(semantic_report_file, 'w', encoding='utf-8') as f:
            f.write("SEMANTIC ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Get WriterService statistics
            stats = writer_service.get_statistics()
            
            f.write("SEMANTIC STORE STATISTICS:\n")
            f.write(f"- Total chunks: {stats['total_chunks']}\n")
            f.write(f"- FAISS vectors: {stats['faiss'].get('total_vectors', 0)}\n")
            f.write(f"- Embedding model: {stats['embeddings'].get('model', 'unknown')}\n")
            f.write(f"- Storage size: {stats['storage'].get('storage_size_mb', 0):.2f} MB\n\n")
            
            # Analyze semantic clusters
            f.write("SEMANTIC CLUSTERING ANALYSIS:\n")
            cluster_analysis = self._analyze_semantic_clusters(writer_service, chunks)
            f.write(cluster_analysis)
    
    def _analyze_semantic_clusters(self, writer_service: WriterService, chunks: List[ChunkData]) -> str:
        """Analyze semantic clusters in the document"""
        analysis_lines = []
        
        # Sample analysis of high-similarity pairs
        high_similarity_pairs = []
        
        for chunk in chunks[:20]:  # Analyze first 20 chunks to avoid overload
            if not chunk.content:
                continue
            
            similar_results = writer_service.search_chunks_semantic(
                query=chunk.content[:200],
                max_results=3,
                min_similarity=0.7
            )
            
            for result in similar_results:
                if result.chunk.id != chunk.id:
                    high_similarity_pairs.append((
                        chunk.hierarchical_id,
                        result.chunk.hierarchical_id,
                        result.similarity_score
                    ))
        
        if high_similarity_pairs:
            analysis_lines.append("HIGH SIMILARITY PAIRS (>0.7):")
            for pair in sorted(high_similarity_pairs, key=lambda x: x[2], reverse=True)[:10]:
                analysis_lines.append(f"- {pair[0]} ↔ {pair[1]} (similarity: {pair[2]:.2f})")
        else:
            analysis_lines.append("No high-similarity pairs found - good semantic diversity.")
        
        return "\n".join(analysis_lines)
    
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