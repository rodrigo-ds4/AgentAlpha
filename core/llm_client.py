"""
Multi-LLM Client supporting Ollama, OpenAI, and DeepSeek
Made by Rodrigo de Sarasqueta
"""

import httpx
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.language_models import BaseLLM


class LLMClient:
    """Unified client for different LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm: Optional[BaseLLM] = None
        self.current_provider = None
        
    async def initialize(self):
        """Initialize the LLM client based on configuration"""
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "ollama")
        
        print(f"Initializing LLM client: {provider}")
        
        if provider == "ollama":
            await self._init_ollama(llm_config)
        elif provider == "openai":
            await self._init_openai(llm_config)
        elif provider == "deepseek":
            await self._init_deepseek(llm_config)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        self.current_provider = provider
        print(f"LLM client initialized: {provider}")
    
    async def _init_ollama(self, config: Dict[str, Any]):
        """Initialize Ollama client"""
        base_url = config.get("ollama_url", "http://localhost:11434")
        model = config.get("ollama_model", "llama3:latest")
        
        # Test connection
        await self._test_ollama_connection(base_url)
        
        self.llm = Ollama(
            base_url=base_url,
            model=model,
            timeout=config.get("timeout", 60)
        )
    
    async def _init_openai(self, config: Dict[str, Any]):
        """Initialize OpenAI client"""
        api_key = config.get("openai_api_key")
        if not api_key:
            raise ValueError("OpenAI API key not found in config")
        
        model = config.get("openai_model", "gpt-3.5-turbo")
        
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1000)
        )
    
    async def _init_deepseek(self, config: Dict[str, Any]):
        """Initialize DeepSeek client (OpenAI-compatible API)"""
        api_key = config.get("deepseek_api_key")
        if not api_key:
            raise ValueError("DeepSeek API key not found in config")
        
        base_url = config.get("deepseek_url", "https://api.deepseek.com")
        model = config.get("deepseek_model", "deepseek-chat")
        
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=config.get("temperature", 0.7),
            max_tokens=config.get("max_tokens", 1000)
        )
    
    async def _test_ollama_connection(self, base_url: str):
        """Test connection to Ollama server"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/api/tags", timeout=10)
                response.raise_for_status()
                
                models = response.json().get("models", [])
                print(f"Connected to Ollama. Available models: {len(models)}")
                
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            raise ConnectionError(f"Cannot connect to Ollama at {base_url}: {e}")
    
    def get_llm(self) -> BaseLLM:
        """Get the initialized LLM instance"""
        if not self.llm:
            raise RuntimeError("LLM client not initialized")
        return self.llm
    
    def get_provider(self) -> str:
        """Get current LLM provider name"""
        return self.current_provider or "unknown"
