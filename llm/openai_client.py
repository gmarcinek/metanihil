import os
from typing import Optional, List

from openai import OpenAI

from .base import BaseLLMClient, LLMConfig
from .models import ModelProvider, MODEL_MAX_TOKENS

class OpenAIClient(BaseLLMClient):
    """Klient dla modeli OpenAI - tylko elite models"""
    
    def __init__(self, model: str):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Brak zmiennej środowiskowej OPENAI_API_KEY")
    
    def chat(self, prompt: str, config: LLMConfig, images: Optional[List[str]] = None) -> str:
        """Wyślij prompt do OpenAI (text lub vision)"""
        try:
            # Handle None max_tokens with model-specific fallback
            max_tokens = config.max_tokens or MODEL_MAX_TOKENS[self.model]
            
            # Przygotuj messages
            if images:
                # Vision mode - OpenAI format
                content = [{"type": "text", "text": prompt}]
                for image_base64 in images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    })
                messages = [{"role": "user", "content": content}]
            else:
                # Text-only mode
                messages = [{"role": "user", "content": prompt}]
            
            # Dodaj system message jeśli istnieje
            if config.system_message:
                messages.insert(0, {"role": "system", "content": config.system_message})
            
            # Przygotuj parametry
            params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": config.temperature,
            }
            
            # Dodaj dodatkowe parametry
            if config.extra_params:
                supported_params = ['top_p', 'frequency_penalty', 'presence_penalty', 'stop', 'seed']
                for key, value in config.extra_params.items():
                    if key in supported_params:
                        params[key] = value
            
            response = self.client.chat.completions.create(**params)
            
            if not response.choices:
                raise RuntimeError("Brak odpowiedzi z OpenAI (choices == []).")
            
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise RuntimeError("OpenAI zwrócił pustą odpowiedź.")
            
            return content.strip()
            
        except Exception as e:
            raise RuntimeError(f"❌ Błąd OpenAI ({self.model}): {e}")
    
    def get_provider(self) -> ModelProvider:
        """Zwróć providera"""
        return ModelProvider.OPENAI