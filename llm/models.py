# PLIK: llm/models.py
from enum import Enum

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"

class Models:
    # OpenAI - emergency backup
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    
    # Anthropic - scenario powerhouse  
    CLAUDE_4_SONNET = "claude-4-sonnet"
    CLAUDE_4_OPUS = "claude-4-opus"
    CLAUDE_3_5_SONNET = "claude-3.5-sonnet"
    CLAUDE_3_5_HAIKU = "claude-3.5-haiku"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    
    # Ollama - coding beasts
    QWEN_CODER = "qwen2.5-coder"
    QWEN_CODER_32B = "qwen2.5-coder:32b" 
    CODESTRAL = "codestral"
    
    # Ollama - vision models
    LLAMA_VISION_11B = "llama3.2-vision:11b"
    LLAMA_VISION_90B = "llama3.2-vision:90b"
    QWEN_VISION_7B = "qwen2.5vl:7b"
    GEMMA3_12B = "gemma3:12b"

# Mapowanie modeli na providerów
MODEL_PROVIDERS = {
    # OpenAI text models
    Models.GPT_4_1_MINI: ModelProvider.OPENAI,
    Models.GPT_4_1_NANO: ModelProvider.OPENAI,
    Models.GPT_4O: ModelProvider.OPENAI,
    Models.GPT_4O_MINI: ModelProvider.OPENAI,
    
    # Anthropic text models
    Models.CLAUDE_4_SONNET: ModelProvider.ANTHROPIC,
    Models.CLAUDE_4_OPUS: ModelProvider.ANTHROPIC,
    Models.CLAUDE_3_5_SONNET: ModelProvider.ANTHROPIC,
    Models.CLAUDE_3_5_HAIKU: ModelProvider.ANTHROPIC,
    Models.CLAUDE_3_HAIKU: ModelProvider.ANTHROPIC,
    
    # Ollama text models
    Models.QWEN_CODER: ModelProvider.OLLAMA,
    Models.QWEN_CODER_32B: ModelProvider.OLLAMA,
    Models.CODESTRAL: ModelProvider.OLLAMA,
    
    # Ollama vision models
    Models.LLAMA_VISION_11B: ModelProvider.OLLAMA,
    Models.LLAMA_VISION_90B: ModelProvider.OLLAMA,
    Models.QWEN_VISION_7B: ModelProvider.OLLAMA,
    Models.GEMMA3_12B: ModelProvider.OLLAMA,
}

# Vision models mapping - FIXED: Added Claude models
VISION_MODELS = {
    # Ollama vision
    Models.LLAMA_VISION_11B: ModelProvider.OLLAMA,
    Models.LLAMA_VISION_90B: ModelProvider.OLLAMA,
    Models.QWEN_VISION_7B: ModelProvider.OLLAMA,
    Models.GEMMA3_12B: ModelProvider.OLLAMA,
    
    # OpenAI vision
    Models.GPT_4O: ModelProvider.OPENAI,
    Models.GPT_4O_MINI: ModelProvider.OPENAI,
    
    # Anthropic vision - ADDED: All Claude models support vision
    Models.CLAUDE_4_SONNET: ModelProvider.ANTHROPIC,
    Models.CLAUDE_4_OPUS: ModelProvider.ANTHROPIC,
    Models.CLAUDE_3_5_SONNET: ModelProvider.ANTHROPIC,
    Models.CLAUDE_3_5_HAIKU: ModelProvider.ANTHROPIC,
    Models.CLAUDE_3_HAIKU: ModelProvider.ANTHROPIC,
}

# Maksymalne limity OUTPUT tokenów
MODEL_MAX_TOKENS = {
    # OpenAI
    Models.GPT_4_1_MINI: 32768,
    Models.GPT_4_1_NANO: 32768,
    Models.GPT_4O: 16384,
    Models.GPT_4O_MINI: 16384,
    
    # Anthropic
    Models.CLAUDE_4_SONNET: 64000,
    Models.CLAUDE_4_OPUS: 32000,
    Models.CLAUDE_3_5_SONNET: 8192,
    Models.CLAUDE_3_5_HAIKU: 8192,
    Models.CLAUDE_3_HAIKU: 4096,        # FIXED: Claude 3 Haiku ma limit 4096!
    
    # Ollama text
    Models.QWEN_CODER: 32768,
    Models.QWEN_CODER_32B: 32768,
    Models.CODESTRAL: 32768,
    Models.GEMMA3_12B: 8192,
    
    # Ollama vision
    Models.LLAMA_VISION_11B: 32768,
    Models.LLAMA_VISION_90B: 32768,
    Models.QWEN_VISION_7B: 128000,
}

# INPUT context window limits
MODEL_INPUT_CONTEXT = {
    # OpenAI
    Models.GPT_4_1_MINI: 1000000,
    Models.GPT_4_1_NANO: 1047576,
    Models.GPT_4O: 128000,
    Models.GPT_4O_MINI: 128000,
    
    # Anthropic
    Models.CLAUDE_4_SONNET: 200000,
    Models.CLAUDE_4_OPUS: 200000,
    Models.CLAUDE_3_5_SONNET: 200000,
    Models.CLAUDE_3_5_HAIKU: 200000,
    Models.CLAUDE_3_HAIKU: 200000,
    
    # Ollama text
    Models.QWEN_CODER: 32768,
    Models.QWEN_CODER_32B: 32768,
    Models.CODESTRAL: 32768,
    
    # Ollama vision
    Models.LLAMA_VISION_11B: 128000,
    Models.LLAMA_VISION_90B: 128000,
    Models.QWEN_VISION_7B: 32768,
    Models.GEMMA3_12B: 128000,
}

def get_model_input_limit(model_name: str) -> int:
    """Get input context window limit for model"""
    return MODEL_INPUT_CONTEXT.get(model_name, 32768)  # safe fallback

def get_model_output_limit(model_name: str) -> int:
    """Get output tokens limit for model"""
    return MODEL_MAX_TOKENS.get(model_name, 8192)  # safe fallback

def supports_vision(model_name: str) -> bool:
    """Check if model supports vision"""
    return model_name in VISION_MODELS

def get_model_provider(model_name: str) -> ModelProvider:
    """Get provider for model"""
    return MODEL_PROVIDERS.get(model_name)