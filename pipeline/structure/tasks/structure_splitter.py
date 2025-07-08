import luigi
import json
import fitz
import re
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from components.structured_task import StructuredTask
from pipeline.structure.tasks.semantic_structure_detector import SemanticStructureDetector
from pipeline.structure.config import get_splitter_config


class StructureSplitter(StructuredTask):
    """Split document based on semantic sections from SemanticStructureDetector"""
    
    file_path = luigi.Parameter()
    
    @property
    def pipeline_name(self) -> str:
        return "structure"
    
    @property
    def task_name(self) -> str:
        return "structure_splitter"
    
    def requires(self):
        return SemanticStructureDetector(file_path=self.file_path)
    
    def run(self):
        print("✂️ Starting structure-based document splitting...")
        
        # Load semantic structure detection results
        with self.input().open('r') as f:
            structure_data = json.load(f)
        
        if structure_data.get("status") != "success":
            raise ValueError("Semantic structure detection failed")
        
        # Load config
        config = get_splitter_config()
        
        # Split document based on detected sections
        sections = self._split_by_semantic_sections(structure_data, config)
        
        # Create output
        result = {
            "task_name": "StructureSplitter",
            "input_file": str(self.file_path),
            "status": "success",
            "sections_created": len(sections),
            "sections": sections,
            "splitting_method": "semantic_structure_based",
            "created_at": datetime.now().isoformat()
        }
        
        with self.output().open('w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Document split into {len(sections)} sections")
    
    def _split_by_semantic_sections(self, structure_data, config):
        """Split document based on semantic sections"""
        semantic_sections = structure_data.get("sections", [])
        
        if not semantic_sections:
            print("⚠️ No semantic sections found for splitting")
            return []
        
        print(f"📊 Found {len(semantic_sections)} semantic sections to process")
        
        # Create output directory
        doc_name = Path(self.file_path).stem
        output_base = Path("output") / doc_name / "structure_sections"
        output_base.mkdir(parents=True, exist_ok=True)
        
        sections = []
        doc = fitz.open(self.file_path)
        
        try:
            # Create PDF sections for each semantic section
            for i, semantic_section in enumerate(semantic_sections):
                section = self._create_section_from_semantic_section(
                    doc, semantic_section, i, output_base, config
                )
                if section:
                    sections.append(section)
                    print(f"📄 Created: {section['filename']}")
        
        finally:
            doc.close()
        
        return sections
    
    def _create_section_from_semantic_section(self, doc, semantic_section, section_index, output_base, config):
        """Create PDF section from semantic section"""
        start_page = semantic_section["start_page"]  # 1-indexed
        end_page = semantic_section["end_page"]      # 1-indexed
        title = semantic_section["title"]
        section_type = semantic_section["section_type"]
        
        # Create filename
        safe_title = self._sanitize_filename(title, config.get("max_filename_length", 50))
        filename = f"section_{section_index:03d}_{section_type}_{safe_title}.pdf"
        file_path = output_base / filename
        
        # Create section PDF with page range
        success = self._extract_page_range_pdf(doc, start_page, end_page, file_path)
        
        if success:
            return {
                "title": title,
                "filename": filename,
                "file_path": str(file_path),
                "start_page": start_page,
                "end_page": end_page,
                "section_type": section_type,
                "section_index": section_index,
                "page_count": end_page - start_page + 1,
                "splitting_method": "semantic_page_range"
            }
        
        return None
    
    def _extract_page_range_pdf(self, source_doc, start_page, end_page, output_path):
        """Extract page range as PDF"""
        try:
            # Convert to 0-indexed for fitz
            start_idx = start_page - 1
            end_idx = end_page - 1
            
            if start_idx >= len(source_doc) or end_idx >= len(source_doc):
                print(f"⚠️ Page range {start_page}-{end_page} exceeds document length ({len(source_doc)})")
                return False
            
            if start_idx > end_idx:
                print(f"⚠️ Invalid page range: {start_page}-{end_page}")
                return False
            
            # Create new PDF with page range
            section_doc = fitz.open()
            
            for page_idx in range(start_idx, end_idx + 1):
                source_page = source_doc[page_idx]
                
                # Copy page to new document
                new_page = section_doc.new_page(
                    width=source_page.rect.width,
                    height=source_page.rect.height
                )
                new_page.show_pdf_page(new_page.rect, source_doc, page_idx)
            
            print(f"🔪 Extracting pages {start_page}-{end_page} ({end_page - start_page + 1} pages)")
            
            # Save section
            section_doc.save(str(output_path), garbage=3, clean=True)
            section_doc.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create section: {e}")
            return False
    
    def _sanitize_filename(self, title, max_length):
        """Sanitize title for filename"""
        # Replace problematic characters
        safe_title = re.sub(r'[<>:"/\\|?*\s]', '_', title)
        safe_title = re.sub(r'_+', '_', safe_title)
        safe_title = safe_title.strip('._')
        
        # Check for empty or only underscores
        if not safe_title or re.fullmatch(r'_+', safe_title):
            safe_title = "untitled"
        
        # Truncate if too long
        if len(safe_title) > max_length:
            safe_title = safe_title[:max_length].rstrip('_')
        
        return safe_title