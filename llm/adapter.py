# llm/adapter.py

from typing import Dict, Any, Optional, List
from .base import LLMConfig, BaseLLMClient
from .models import ModelProvider, MODEL_PROVIDERS, MODEL_MAX_TOKENS, VISION_MODELS
from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient
from .ollama_client import OllamaClient
from .utils import log_llm_request, log_llm_response


class LLMClient:
    """Minimalistyczny adapter zarządzający różnymi providerami LLM z vision support"""
    
    def __init__(self, model: str, max_tokens: Optional[int] = None, temperature: float = 0.0, 
                 system_message: Optional[str] = None, fresh_client_every_request: bool = True):
        """
        Inicjalizuj klienta LLM
        
        Args:
            model: Nazwa modelu (np. Models.CLAUDE_4_SONNET, Models.QWEN_CODER_32B)
            max_tokens: Maksymalna liczba tokenów (None = użyj maksimum dla modelu)
            temperature: Temperatura modelu (0.0-1.0)
            system_message: Opcjonalny system message
            fresh_client_every_request: Create fresh client each request (prevents context bleeding)
        """
        if model not in MODEL_PROVIDERS:
            raise ValueError(f"Nieobsługiwany model: {model}. Dostępne: {list(MODEL_PROVIDERS.keys())}")
        
        self.model = model
        self.provider = MODEL_PROVIDERS[model]
        self.fresh_every_request = fresh_client_every_request
        
        # Ustaw max_tokens - użyj maksimum dla modelu jeśli nie podano
        if max_tokens is None:
            max_tokens = MODEL_MAX_TOKENS[self.model]
        
        self.config = LLMConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            system_message=system_message
        )
        
        # Zainicjalizuj odpowiedni klient
        self.client = self._create_client()
    
    def _create_client(self) -> BaseLLMClient:
        """Utwórz odpowiedni klient na podstawie providera"""
        if self.provider == ModelProvider.OPENAI:
            return OpenAIClient(self.model)
        elif self.provider == ModelProvider.ANTHROPIC:
            return AnthropicClient(self.model)
        elif self.provider == ModelProvider.OLLAMA:
            return OllamaClient(self.model)
        else:
            raise ValueError(f"Nieobsługiwany provider: {self.provider}")
    
    def chat(self, prompt: str, config: Optional[LLMConfig] = None, images: List[str] = None) -> str:
        """
        Wyślij prompt do modelu (z opcjonalnymi obrazkami)
        
        Args:
            prompt: Tekst zapytania
            config: Opcjonalna konfiguracja (nadpisuje domyślną)
            images: Lista base64 obrazków (dla vision models)
        """
        use_config = config if config else self.config
        
        # TYLKO TO: Fresh client every request
        if self.fresh_every_request:
            self.client = self._create_client()
        
        # LOG REQUEST
        log_llm_request(prompt, use_config, self.model)
        
        # Vision vs text routing
        if images:
            if not self._supports_vision():
                raise ValueError(f"Model {self.model} doesn't support vision")
            response = self.client.chat(prompt, use_config, images)  # Unified method
        else:
            response = self.client.chat(prompt, use_config)
        
        # LOG RESPONSE  
        log_llm_response(prompt, response, use_config, self.model)
        
        return response
    
    def clear_context(self):
        """Manually clear context - useful for debugging"""
        if hasattr(self.client, 'clear_context'):
            self.client.clear_context()
        else:
            # Fallback: recreate client
            self.client = self._create_client()
    
    def _supports_vision(self) -> bool:
        """Check if current model supports vision"""
        return self.model in VISION_MODELS
    
    def get_max_tokens_for_model(self) -> int:
        """Zwróć maksymalną liczbę tokenów dla bieżącego modelu"""
        return MODEL_MAX_TOKENS[self.model]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Zwróć informacje o modelu"""
        return {
            "model": self.model,
            "provider": self.provider.value,
            "max_tokens_available": MODEL_MAX_TOKENS[self.model],
            "supports_vision": self._supports_vision(),
            "fresh_every_request": self.fresh_every_request,
            "current_config": {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "has_system_message": bool(self.config.system_message)
            }
        }