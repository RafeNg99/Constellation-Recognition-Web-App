import os

# Ollama API URL
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
URL = f"{OLLAMA_BASE_URL}/api/chat"

# LLM model name
MODEL_NAME = "qwen3:4b"
