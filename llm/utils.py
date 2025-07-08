# llm/utils.py

from typing import Optional
from datetime import datetime
from pathlib import Path
from .base import LLMConfig
from .json_utils import parse_json_with_markdown_blocks  # ← IMPORT Z NOWEGO PLIKU


def detect_image_format(image_base64: str) -> str:
    """
    Auto-detect image format from base64 data
    
    Args:
        image_base64: Base64 encoded image data
        
    Returns:
        MIME type string (image/jpeg, image/png, etc.)
    """
    try:
        # Check base64 signature patterns
        if image_base64.startswith("/9j/"):
            return "image/jpeg"
        elif image_base64.startswith("iVBOR"):
            return "image/png"
        elif image_base64.startswith("R0lGOD"):
            return "image/gif"
        elif image_base64.startswith("UklGR"):
            return "image/webp"
        else:
            # Default fallback - most PDF extractions are JPEG
            return "image/jpeg"
    except Exception:
        # Safe fallback
        return "image/jpeg"


def log_llm_request(prompt: str, config: LLMConfig, model: str):
    """Log LLM request to file"""
    try:
        logs_dir = Path("semantic_store/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"llm_{timestamp}_REQUEST.txt"
        
        log_content = f"""=== LLM REQUEST ===
Timestamp: {datetime.now().isoformat()}
Model: {model}
Temperature: {config.temperature}
Max Tokens: {config.max_tokens}
System Message: {config.system_message or 'None'}

PROMPT:
{prompt}
"""
        
        (logs_dir / filename).write_text(log_content, encoding='utf-8')
        
    except Exception as e:
        print(f"⚠️ Failed to log LLM request: {e}")


def log_llm_response(prompt: str, response: str, config: LLMConfig, model: str):
    """Log LLM response to file"""
    try:
        logs_dir = Path("semantic_store/logs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"llm_{timestamp}_RESPONSE.txt"
        
        log_content = f"""=== LLM RESPONSE ===
Timestamp: {datetime.now().isoformat()}
Model: {model}
Prompt Length: {len(prompt)} chars
Response Length: {len(response)} chars
Response Word Count: {len(response.split())}

RESPONSE:
{response}
"""
        
        (logs_dir / filename).write_text(log_content, encoding='utf-8')
        
    except Exception as e:
        print(f"⚠️ Failed to log LLM response: {e}")


# Convenience functions using enterprise parser
def get_json_parsing_stats():
    """Get JSON parsing statistics - convenience function"""
    from .json_utils import get_json_parsing_stats
    return get_json_parsing_stats()


def reset_json_parsing_stats():
    """Reset JSON parsing statistics - convenience function"""
    from .json_utils import reset_json_parsing_stats
    reset_json_parsing_stats()