import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    def __init__(self, config_path: str = "pipeline/writer/config.yaml"):
        self.config_path = config_path
        self._config = None
    
    @property
    def config(self) -> Dict[str, Any]:
        """Load and cache config"""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> Dict[str, Any]:
        """Load config from YAML file"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get_task_config(self, task_name: str) -> Dict[str, Any]:
        """Get configuration for specific task"""
        tasks_config = self.config.get('tasks', {})
        return tasks_config.get(task_name, {})
    
    def get_database_path(self) -> str:
        """Get database file path"""
        return self.config['database']['path']
    
    def get_progress_file(self) -> str:
        """Get progress tracking file path"""
        return self.config['progress']['file']
    
    def get_output_config(self) -> Dict[str, str]:
        """Get output configuration"""
        return self.config['output']
    
    def get_cache_config(self) -> Dict[str, str]:
        """Get cache configuration"""
        return self.config['cache']