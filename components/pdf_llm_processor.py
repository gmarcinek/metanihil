import asyncio
from dotenv import load_dotenv
load_dotenv()

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import fitz
import base64
from pathlib import Path

from llm import LLMClient, LLMConfig


@dataclass
class ProcessingConfig:
    """Config for PDF → LLM processing"""
    # Required fields - no defaults
    model: str
    prompt_template: str
    
    # Optional fields - with defaults (single source of truth)
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_concurrent: int = 2
    rate_limit_backoff: float = 30.0
    target_width_px: int = 800
    jpg_quality: int = 65
    clean_text: bool = False  # NEW: text cleaning flag
    
    @classmethod
    def from_yaml(cls, yaml_dict: dict, task_name: str = "Unknown"):
        """Create config from YAML dict - FAIL FAST on missing values"""
        
        # Check if we got any config at all
        if not yaml_dict:
            print(f"❌ FATAL: Task '{task_name}' passed EMPTY config to PDFLLMProcessor")
            raise ValueError(f"Task '{task_name}' must provide YAML config section")
        
        # Required fields - WYJEB if missing
        required_fields = {
            "model": "LLM model name (e.g. 'claude-3.5-haiku')",
            "vision_prompt": "LLM prompt template with {text_content} placeholder"
        }
        
        missing_fields = []
        for field, description in required_fields.items():
            if field not in yaml_dict:
                print(f"❌ MISSING REQUIRED: '{field}' - {description}")
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ FATAL: Task '{task_name}' config missing: {missing_fields}")
            raise ValueError(f"Required config fields missing: {missing_fields}")
        
        # Check for empty required values
        if not yaml_dict["vision_prompt"].strip():
            print(f"❌ FATAL: Task '{task_name}' has EMPTY vision_prompt")
            raise ValueError("vision_prompt cannot be empty")
        
        print(f"✅ Config loaded for task '{task_name}': {yaml_dict['model']}")
        
        # Create instance using YAML values OR dataclass defaults
        return cls(
            # Required fields from YAML
            model=yaml_dict["model"],
            prompt_template=yaml_dict["vision_prompt"],
            
            # Optional fields - YAML overrides dataclass defaults
            temperature=yaml_dict.get("temperature", cls.temperature),
            max_tokens=yaml_dict.get("max_tokens", cls.max_tokens),
            max_concurrent=yaml_dict.get("max_concurrent", cls.max_concurrent),
            rate_limit_backoff=yaml_dict.get("rate_limit_backoff", cls.rate_limit_backoff),
            target_width_px=yaml_dict.get("target_width_px", cls.target_width_px),
            jpg_quality=yaml_dict.get("jpg_quality", cls.jpg_quality),
            clean_text=yaml_dict.get("clean_text", cls.clean_text)
        )


@dataclass
class PageResult:
    """Result from processing single page"""
    page_num: int
    status: str  # "success" or "error"
    result: Dict[str, Any] = None
    error: str = ""
    retry_after_rate_limit: bool = False


class SlidingWindowPageProcessor:
    """Single page processor for sliding window"""
    
    def __init__(self, page_data: Dict, config: ProcessingConfig, 
                 response_parser: Callable[[str], Dict[str, Any]] = None):
        self.page_data = page_data
        self.config = config
        self.response_parser = response_parser
        self.llm_client = LLMClient(config.model)
        self.llm_config = LLMConfig(
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    def process_page(self) -> PageResult:
        """Process single page and return result"""
        page_num = self.page_data["page_num"]
        
        try:
            result = self._call_llm_vision()
            
            return PageResult(
                page_num=page_num,
                status="success",
                result=result
            )
            
        except Exception as e:
            # Rate limit handling
            if self._is_rate_limit_error(e):
                print(f"🚨 Rate limit detected! Waiting {self.config.rate_limit_backoff}s...")
                time.sleep(self.config.rate_limit_backoff)
                
                # Try once more after backoff
                try:
                    result = self._call_llm_vision()
                    
                    return PageResult(
                        page_num=page_num,
                        status="success",
                        result=result,
                        retry_after_rate_limit=True
                    )
                except Exception as retry_e:
                    print(f"❌ Retry also failed for page {page_num}: {retry_e}")
                    return PageResult(
                        page_num=page_num,
                        status="error",
                        error=f"Rate limit retry failed: {retry_e}"
                    )
            
            return PageResult(
                page_num=page_num,
                status="error",
                error=str(e)
            )
    
    def _call_llm_vision(self) -> Dict[str, Any]:
        """Call LLM with vision - core logic"""
        text = self.page_data.get("text", "")
        image_base64 = self.page_data.get("image_base64")
        
        # Build prompt
        prompt = self.config.prompt_template.replace("{text_content}", text)
        
        # Call LLM with image
        response = self.llm_client.chat(prompt, self.llm_config, images=[image_base64])
        
        # Parse response
        if self.response_parser:
            parsed_result = self.response_parser(response)
            if parsed_result is None:
                raise ValueError("Failed to parse LLM response")
            return parsed_result
        else:
            return {"raw_response": response}
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Detect rate limit errors"""
        error_str = str(error).lower()
        return any(phrase in error_str for phrase in [
            "rate limit",
            "429",
            "too many requests",
            "rate_limit_exceeded"
        ])


class PDFLLMProcessor:
    """Generic PDF → screenshots + text → LLM processor with sliding window"""
    
    def __init__(self, yaml_config: dict, task_name: str = "Unknown"):
        """Initialize with YAML config - FAIL FAST if config invalid"""
        try:
            self.config = ProcessingConfig.from_yaml(yaml_config, task_name)
            print(f"🚀 PDFLLMProcessor ready for task '{task_name}'")
        except Exception as e:
            print(f"💥 PDFLLMProcessor FAILED to initialize for task '{task_name}': {e}")
            raise
    
    def process_pdf(self, pdf_path: str, 
                   response_parser: Callable[[str], Dict[str, Any]] = None) -> List[PageResult]:
        """
        Process PDF pages through LLM using sliding window
        
        Args:
            pdf_path: Path to PDF file
            response_parser: Function to parse LLM response (optional)
            
        Returns:
            List of PageResult objects
        """
        print(f"🚀 Starting PDF processing: {Path(pdf_path).name}")
        
        # Extract pages with text + images
        pages_data = self._extract_pages_data(pdf_path)
        
        if not pages_data:
            print("❌ No pages extracted")
            return []
        
        print(f"📊 Processing {len(pages_data)} pages with sliding window (max_concurrent={self.config.max_concurrent})")
        
        # Process with sliding window
        results = asyncio.run(self._process_with_sliding_window(pages_data, response_parser))
        
        # Summary
        success_count = sum(1 for r in results if r.status == "success")
        print(f"✅ Completed: {success_count}/{len(results)} pages successful")
        
        return results
    
    def _extract_pages_data(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text + images from all PDF pages"""
        doc = fitz.open(pdf_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                
                # Extract text
                text = page.get_text().strip() or "[No text extracted]"
                
                # Clean text if enabled
                if self.config.clean_text:
                    print(f"🔍 RAW text page {page_num + 1} ({len(text)} chars): {repr(text[:100])}")
                    text = self._clean_extracted_text(text)
                    print(f"🧹 CLEANED text page {page_num + 1} ({len(text)} chars): {repr(text[:100])}")
                
                # Create screenshot
                image_base64 = self._create_screenshot(page)
                
                pages_data.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "image_base64": image_base64
                })
                
            except Exception as e:
                print(f"❌ Failed to extract page {page_num + 1}: {e}")
                continue
        
        doc.close()
        return pages_data
    
    def _create_screenshot(self, page) -> str:
        """Create base64 screenshot of page"""
        page_rect = page.rect
        zoom = self.config.target_width_px / page_rect.width
        zoom = max(0.5, min(3.0, zoom))
        
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("jpeg", jpg_quality=self.config.jpg_quality)
        
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def _clean_extracted_text(self, raw_text: str) -> str:
        """Clean PDF extraction artifacts for better LLM parsing"""
        import re
        
        # 1. Normalize multiple spaces to single
        cleaned = re.sub(r' {2,}', ' ', raw_text)
        
        # 2. Normalize multiple dots (dot leaders) 
        cleaned = re.sub(r'\.{3,}', '...', cleaned)
        
        # 3. Remove weird Unicode spaces
        cleaned = re.sub(r'[\u00A0\u2000-\u200B\u2028\u2029]', ' ', cleaned)
        
        # 4. Normalize line breaks
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # 5. Strip each line and remove empty lines
        lines = [line.strip() for line in cleaned.split('\n')]
        cleaned = '\n'.join(line for line in lines if line)
        
        return cleaned
    
    async def _process_with_sliding_window(self, pages_data: List[Dict], 
                                         response_parser: Callable) -> List[PageResult]:
        """Process pages with sliding window"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_single_page(page_data: Dict):
            async with semaphore:
                # Create page processor
                page_processor = SlidingWindowPageProcessor(
                    page_data=page_data,
                    config=self.config,
                    response_parser=response_parser
                )
                
                # Run in thread pool (LLM calls are blocking)
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(executor, page_processor.process_page)
                
                status = "✅" if result.status == "success" else "❌"
                retry_info = " (after rate limit retry)" if result.retry_after_rate_limit else ""
                print(f"{status} Page {result.page_num} completed{retry_info}")
                
                return result
        
        # Process all pages concurrently with sliding window
        tasks = [process_single_page(page_data) for page_data in pages_data]
        results = await asyncio.gather(*tasks)
        
        return results