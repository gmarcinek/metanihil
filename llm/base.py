from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from .models import ModelProvider


@dataclass
class LLMConfig:
    """Konfiguracja dla modeli LLM"""
    max_tokens: Optional[int] = None
    temperature: float = 0.0
    system_message: Optional[str] = None
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


class BaseLLMClient(ABC):
    """Abstrakcyjna klasa bazowa dla clientów LLM z vision support"""
    
    @abstractmethod
    def chat(self, prompt: str, config: LLMConfig, images: Optional[List[str]] = None) -> str:
        """Wyślij prompt i otrzymaj odpowiedź (z opcjonalnymi obrazkami)"""
        pass
    
    @abstractmethod
    def get_provider(self) -> ModelProvider:
        """Zwróć providera modelu"""
        pass