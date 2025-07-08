import luigi
import json
import fitz
from pathlib import Path
from datetime import datetime
from PIL import Image
import io
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from components.structured_task import StructuredTask
from ocr import SuryaClient


class StructureDetector(StructuredTask):
    file_path = luigi.Parameter()
    
    @property
    def pipeline_name(self) -> str:
        return "structure"
    
    @property
    def task_name(self) -> str:
        return "structure_detector"
    
    def run(self):
        print("🔍 Extracting large text blocks...")
        
        config = self._load_config()
        large_blocks = self._extract_large_blocks(config)
        
        result = {
            "task_name": "StructureDetector",
            "input_file": str(self.file_path),
            "status": "success",
            "large_blocks": large_blocks,
            "blocks_count": len(large_blocks),
            "config_used": config,
            "created_at": datetime.now().isoformat()
        }
        
        with self.output().open('w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Found {len(large_blocks)} large text blocks (min_area: {config['min_block_area']})")
    
    def _load_config(self):
        """Load config from YAML - nie hardkoduj gówna"""
        try:
            import yaml
            config_file = Path(__file__).parent.parent / "config.yaml"
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            return config["StructureDetector"]
            
        except FileNotFoundError:
            raise RuntimeError(f"Config file not found: {config_file}")
        except KeyError:
            raise RuntimeError("StructureDetector section missing in config.yaml")
        except Exception as e:
            raise RuntimeError(f"Config load failed: {e}")
    
    def _extract_large_blocks(self, config):
        """Extract duże bloki tekstu z PDF z merge logic"""
        doc = fitz.open(self.file_path)
        max_pages = min(len(doc), config.get("max_pages", 1000))
        
        # Convert pages to images with higher resolution
        images = []
        for page_num in range(max_pages):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(config["zoom_factor"], config["zoom_factor"]))
            img_bytes = pix.tobytes("png")
            images.append(Image.open(io.BytesIO(img_bytes)))
        
        doc.close()
        
        # Use Surya layout detection
        surya_client = SuryaClient()
        results = surya_client.process_pages(images)
        
        # Extract ALL blocks first (nie filtruj po area)
        all_blocks = []
        for page_idx, result in enumerate(results):
            layout = result.get('layout', [])
            
            # Handle both dict and LayoutResult objects
            layout_elements = []
            if hasattr(layout, 'bboxes'):
                # LayoutResult object
                layout_elements = layout.bboxes
            elif isinstance(layout, list):
                # List of elements
                layout_elements = layout
            else:
                print(f"⚠️ Unknown layout type: {type(layout)}")
                continue
            
            for element in layout_elements:
                # Handle both dict and bbox objects
                if hasattr(element, 'bbox'):
                    bbox = element.bbox
                    label = getattr(element, 'label', 'unknown')
                    confidence = getattr(element, 'confidence', 0.0)
                elif isinstance(element, dict):
                    bbox = element.get('bbox', [])
                    label = element.get('label', 'unknown')
                    confidence = element.get('confidence', 0.0)
                else:
                    print(f"⚠️ Unknown element type: {type(element)}")
                    continue
                
                if len(bbox) >= 4:
                    height = bbox[3] - bbox[1]
                    width = bbox[2] - bbox[0]
                    area = height * width
                    
                    # Dodaj WSZYSTKIE bloki (nie filtruj jeszcze)
                    all_blocks.append({
                        "page": page_idx + 1,
                        "bbox": bbox,
                        "width": width,
                        "height": height,
                        "area": area,
                        "label": label,
                        "confidence": confidence
                    })
        
        print(f"📊 Found {len(all_blocks)} raw blocks before merging")
        
        # MERGE consecutive blocks
        merged_blocks = self._merge_consecutive_blocks(all_blocks, config)
        
        print(f"📊 After merging: {len(merged_blocks)} blocks")
        
        # Filter by area AFTER merging
        large_blocks = [b for b in merged_blocks if b['area'] > config["min_block_area"]]
        
        # Sort by page and position
        large_blocks.sort(key=lambda b: (b['page'], b['bbox'][1]))
        
        return large_blocks
    
    def _merge_consecutive_blocks(self, blocks, config):
        """Merge consecutive blocks that are close together"""
        if not blocks:
            return []
        
        # Sort by page and Y position
        blocks.sort(key=lambda b: (b['page'], b['bbox'][1]))
        
        merge_gap_px = config.get("merge_gap_px", 50)
        merged = []
        current_group = [blocks[0]]
        
        for block in blocks[1:]:
            last_block = current_group[-1]
            
            # Same page and close vertically
            same_page = block['page'] == last_block['page']
            vertical_gap = block['bbox'][1] - last_block['bbox'][3]
            close_enough = vertical_gap < merge_gap_px
            
            if same_page and close_enough:
                current_group.append(block)
                print(f"🔗 Merging blocks (gap: {vertical_gap:.1f}px)")
            else:
                # Merge current group and start new one
                merged_block = self._merge_block_group(current_group)
                merged.append(merged_block)
                current_group = [block]
        
        # Don't forget last group
        if current_group:
            merged_block = self._merge_block_group(current_group)
            merged.append(merged_block)
        
        return merged
    
    def _merge_block_group(self, group):
        """Merge group of blocks into single block"""
        if len(group) == 1:
            return group[0]
        
        # Calculate merged bbox
        min_x = min(b['bbox'][0] for b in group)
        min_y = min(b['bbox'][1] for b in group)
        max_x = max(b['bbox'][2] for b in group)
        max_y = max(b['bbox'][3] for b in group)
        
        merged_bbox = [min_x, min_y, max_x, max_y]
        merged_width = max_x - min_x
        merged_height = max_y - min_y
        merged_area = merged_width * merged_height
        
        print(f"🎯 Merged {len(group)} blocks → {merged_area:.0f}px² (was: {[b['area'] for b in group]})")
        
        return {
            "page": group[0]['page'],
            "bbox": merged_bbox,
            "width": merged_width,
            "height": merged_height,
            "area": merged_area,
            "label": "merged_text",
            "confidence": sum(b['confidence'] for b in group) / len(group),
            "merged_count": len(group),
            "original_labels": [b['label'] for b in group]
        }