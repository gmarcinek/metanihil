"""
Document endpoints - PyMuPDF with Unicode fonts
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response
import fitz  # PyMuPDF
from pathlib import Path

from writer.models import ChunkStatus


router = APIRouter(prefix="/document", tags=["document"])


def _get_unicode_font(doc):
    """Get Unicode font for PyMuPDF"""
    # Try different font paths
    font_paths = [
        "C:/Windows/Fonts/calibri.ttf",
    ]
    
    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                fontname = doc.insert_font(fontfile=font_path)
                print(f"✅ Using font: {font_path}")
                return fontname
            except Exception as e:
                print(f"⚠️ Failed to load {font_path}: {e}")
                continue
    
    # Fallback to built-in (will have encoding issues)
    print("⚠️ No Unicode font found, using helv (limited Polish support)")
    return "helv"


@router.get("/full", response_class=PlainTextResponse)
async def get_full_document():
    """Plain text document"""
    from api.main import get_writer_service
    service = get_writer_service()
    
    chunks = service.get_chunks_by_status(ChunkStatus.COMPLETED)
    if not chunks:
        raise HTTPException(status_code=404, detail="No completed chunks")
    
    chunks.sort(key=lambda x: x.hierarchical_id)
    
    parts = []
    for chunk in chunks:
        if chunk.level == 1:
            parts.append(f"\n\n{chunk.title.upper()}\n{'='*len(chunk.title)}")
        elif chunk.level == 2:
            parts.append(f"\n\n{chunk.title}\n{'-'*len(chunk.title)}")
        
        if chunk.content:
            parts.append(f"\n\n{chunk.content}")
    
    return "".join(parts).strip()


@router.get("/pdf")
async def get_pdf():
    """PDF document with Unicode support"""
    from api.main import get_writer_service
    service = get_writer_service()
    
    chunks = service.get_chunks_by_status(ChunkStatus.COMPLETED)
    if not chunks:
        raise HTTPException(status_code=404, detail="No completed chunks")
    
    chunks.sort(key=lambda x: x.hierarchical_id)
    
    # Create new PDF document
    doc = fitz.open()
    
    # Load Unicode font
    fontname = _get_unicode_font(doc)
    
    page = None
    y_pos = 50
    
    for chunk in chunks:
        # New page for level 1 chapters
        if chunk.level == 1:
            page = doc.new_page()
            y_pos = 80
            
            # Chapter title
            page.insert_text(
                (50, y_pos),
                chunk.title,
                fontsize=18,
                fontname=fontname,
                color=(0, 0, 0)
            )
            y_pos += 40
            
        elif chunk.level == 2:
            # Section title
            if y_pos > 750:  # New page if near bottom
                page = doc.new_page()
                y_pos = 50
            
            page.insert_text(
                (50, y_pos),
                chunk.title,
                fontsize=14,
                fontname=fontname,
                color=(0, 0, 0)
            )
            y_pos += 25
        
        # Content
        if chunk.content:
            # Split content into paragraphs
            paragraphs = chunk.content.split('\n\n')
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue
                
                # Word wrap
                words = paragraph.strip().split()
                line = ""
                
                for word in words:
                    test_line = line + " " + word if line else word
                    
                    # Simple character-based line length check
                    if len(test_line) > 85:  # chars per line
                        if line:
                            # Insert current line
                            if y_pos > 750:  # New page
                                page = doc.new_page()
                                y_pos = 50
                            
                            page.insert_text(
                                (50, y_pos), 
                                line, 
                                fontsize=11, 
                                fontname=fontname
                            )
                            y_pos += 15
                            line = word
                        else:
                            line = word
                    else:
                        line = test_line
                
                # Insert last line of paragraph
                if line:
                    if y_pos > 750:  # New page
                        page = doc.new_page()
                        y_pos = 50
                    
                    page.insert_text(
                        (50, y_pos), 
                        line, 
                        fontsize=11, 
                        fontname=fontname
                    )
                    y_pos += 15
                
                # Space between paragraphs
                y_pos += 10
    
    # Get PDF bytes
    pdf_bytes = doc.write()
    doc.close()
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=document.pdf"}
    )