"""
Simple configuration management for the minimal agent
Made by Rodrigo de Sarasqueta
"""

import json
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Simple configuration manager"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from JSON file"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print(f"Configuration loaded from {self.config_path}")
            except Exception as e:
                print(f"Error loading config: {e}")
                self.config = self.get_default_config()
        else:
            print("Config file not found, using defaults")
            self.config = self.get_default_config()
            self.save_config()
    
    def save_config(self):
        """Save current configuration to JSON file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "system_instructions": "You are a helpful AI assistant with access to document search and web APIs. Use your tools when needed to provide accurate information.",
            
            "llm": {
                "provider": "ollama",
                "temperature": 0.7,
                "max_tokens": 1000,
                "timeout": 60,
                
                # Ollama settings
                "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
                "ollama_model": os.getenv("OLLAMA_MODEL", "llama3:latest"),
                
                # OpenAI settings
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "openai_model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                
                # DeepSeek settings
                "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY"),
                "deepseek_url": os.getenv("DEEPSEEK_URL", "https://api.deepseek.com"),
                "deepseek_model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            },
            
            "memory": {
                "max_token_limit": 2000
            },
            
            "vector_store": {
                "db_path": "./data/chroma_db",
                "collection_name": "documents",
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            
            "retrieval": {
                "max_results": 5,
                "similarity_threshold": 0.7
            },
            
            "http": {
                "timeout": 30,
                "retry_attempts": 3
            }
        }
    
    def get(self, key: str, default=None):
        """Get configuration value with dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value with dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self.save_config()
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self.config.copy()
