"""
Config loader for structure pipeline - YAGNI version
"""

import yaml
from pathlib import Path


def load_config():
    """Load config or die trying"""
    config_file = Path(__file__).parent / "config.yaml"
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def get_detector_config():
    """Get StructureDetector config"""
    return load_config()["StructureDetector"]


def get_splitter_config():
    """Get StructureSplitter config"""
    return load_config()["StructureSplitter"]