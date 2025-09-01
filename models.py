import os
from langchain_ollama import OllamaEmbeddings, ChatOllama

class Models:
    def __init__(self):
        # Ollama pull mxbai-embed-large:latest
        self.embeddings_ollama = OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large:latest"),
        )
        
        # Ollama pull llama3 (or whatever model you have)
        self.model_ollama = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3:latest"),  # Changed from "llama3:latest"  &  qwen3:0.6b
            temperature=0,
        )