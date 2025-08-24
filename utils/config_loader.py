# utils/config_loader.py
"""Configuration loading and validation utilities."""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    required_sections = ['model', 'interpolation', 'processing', 'output']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")

    # Validate model config
    model_config = config['model']
    device = model_config.get('device', 'auto')
    if device == 'auto':
        model_config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        model_config['device'] = 'cpu'

    # Validate interpolation config
    interp_config = config['interpolation']
    factor = interp_config.get('interpolation_factor', 2)
    if factor < 2 or factor > 8:
        raise ValueError(f"Interpolation factor must be between 2 and 8, got {factor}")

    target_fps = interp_config.get('target_fps')
    if target_fps and (target_fps <= 0 or target_fps > 240):
        raise ValueError(f"Target FPS must be between 1 and 240, got {target_fps}")

    # Validate processing config
    proc_config = config['processing']
    batch_size = proc_config.get('batch_size', 4)
    if batch_size < 1 or batch_size > 32:
        raise ValueError(f"Batch size must be between 1 and 32, got {batch_size}")

    max_resolution = proc_config.get('max_resolution', [1920, 1080])
    if not isinstance(max_resolution, list) or len(max_resolution) != 2:
        raise ValueError("max_resolution must be a list of [width, height]")

    logger.info("Configuration validation passed")


def create_paths_from_config(paths_config: Dict[str, Any]) -> None:
    """Create necessary directories from paths configuration."""
    path_keys = ['input_dir', 'output_dir', 'temp_dir', 'cache_dir', 'log_dir', 'models_dir']

    for key in path_keys:
        if key in paths_config:
            path = Path(paths_config[key])
            path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {path}")