"""
Configuration module for Claims Processing AI
"""

import json
from pathlib import Path
from typing import Dict, Any

class Config:
    """System configuration manager"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                "system": {
                    "offline_mode": True,
                    "data_dir": "data",
                    "models_dir": "models",
                    "chroma_db_path": "data/chroma_db"
                },
                "models": {
                    "text_model": "phi4-mini:3.8b",
                    "embedding_model": "nomic-embed-text",
                    "ollama_host": "http://127.0.0.1:11434"
                },
                "processing": {
                    "chunk_size": 300,
                    "chunk_overlap": 50,
                    "max_context_length": 2500,
                    "temperature": 0.1,
                    "max_retrieval_chunks": 5
                },
                "api": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "reload": False,
                    "max_upload_size": 10 * 1024 * 1024  # 10MB
                }
            }
    
    def get(self, key_path: str, default=None):
        """Get configuration value by dot-separated path"""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

# Global configuration instance
config = Config()