# llm/anthropic_client.py

import os
from typing import Optional, List

from anthropic import Anthropic

from .base import BaseLLMClient, LLMConfig
from .models import ModelProvider, MODEL_MAX_TOKENS
from .utils import detect_image_format


class AnthropicClient(BaseLLMClient):
    """Klient dla Claude 4 - scenario & planning powerhouse"""
    
    # Mapowanie nazw modeli na rzeczywiste nazwy API
    MODEL_MAPPING = {
        # Claude 4 - najnowsza rodzina
        "claude-4-sonnet": "claude-sonnet-4-20250514",
        "claude-4-opus": "claude-opus-4-20250514",
        
        # Claude 3.5 - poprzednia generacja, wciąż bardzo dobra
        "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
        "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        
        # Claude 3 - stabilna, sprawdzona generacja
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229", 
        "claude-3-haiku": "claude-3-haiku-20240307",
        
        # Aliasy dla wygody
        "sonnet": "claude-sonnet-4-20250514",
        "opus": "claude-opus-4-20250514", 
        "haiku": "claude-3-5-haiku-20241022",
        
        # Aliasy wersji
        "latest": "claude-sonnet-4-20250514",
        "fastest": "claude-3-5-haiku-20241022",
        "smartest": "claude-opus-4-20250514",
        "balanced": "claude-sonnet-4-20250514",
    }
    
    def __init__(self, model: str):
        self.model = model
        self._init_client()
        
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("Brak zmiennej środowiskowej ANTHROPIC_API_KEY")
    
    def _init_client(self):
        """Initialize or re-initialize Anthropic client"""
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def clear_context(self):
        """Force clear any cached context by recreating client"""
        print(f"🧹 Clearing Anthropic context for {self.model}")
        self._init_client()
    
    def _get_api_model_name(self) -> str:
        """Zwróć rzeczywistą nazwę modelu dla API"""
        return self.MODEL_MAPPING.get(self.model, self.model)
    
    def chat(self, prompt: str, config: LLMConfig, images: Optional[List[str]] = None) -> str:
        """Wyślij prompt do Claude (text lub vision)"""
        try:
            api_model = self._get_api_model_name()
            
            # Handle None max_tokens with model-specific fallback
            max_tokens = config.max_tokens or MODEL_MAX_TOKENS[self.model]
            
            # Przygotuj messages
            if images:
                # Vision mode - Claude format z obrazkami
                content = [{"type": "text", "text": prompt}]
                for image_base64 in images:
                    # Auto-detect image format using utils
                    media_type = detect_image_format(image_base64)
                    
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_base64
                        }
                    })
                messages = [{"role": "user", "content": content}]
            else:
                # Text-only mode
                messages = [{"role": "user", "content": prompt}]
            
            # Przygotuj parametry - Claude lubi duże limity
            params = {
                "model": api_model,
                "max_tokens": max_tokens,
                "messages": messages,
                "temperature": config.temperature,
                "timeout": 600.0,  # 10 minut timeout
                "stream": False    # Explicit non-streaming
            }
            
            # Dodaj system message jeśli istnieje
            if config.system_message:
                params["system"] = config.system_message
            
            # Dodaj dodatkowe parametry wspierane przez Anthropic
            if config.extra_params:
                supported_params = ['top_p', 'top_k', 'stop_sequences']
                for key, value in config.extra_params.items():
                    if key in supported_params:
                        params[key] = value
            
            response = self.client.messages.create(**params)
            
            if not response.content:
                raise RuntimeError("Brak odpowiedzi z Claude (content == []).")
            
            # Claude zwraca listę bloków treści
            content = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    content += block.text
            
            if not content.strip():
                raise RuntimeError("Claude zwrócił pustą odpowiedź.")
            
            return content.strip()
            
        except Exception as e:
            raise RuntimeError(f"❌ Błąd Claude ({self.model}): {e}")
    
    def get_provider(self) -> ModelProvider:
        """Zwróć providera"""
        return ModelProvider.ANTHROPIC