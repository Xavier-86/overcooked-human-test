#!/usr/bin/env python
"""
Environment-specific configurations
"""

from typing import Dict, Tuple


# Environment layout configurations
ENV_CONFIGS = {
    # Overcooked layouts
    "random1_m": {
        "version": "new",
        "max_players": 2,
        "temporal_state_num": 52,
    },
    "random3_m": {
        "version": "new",
        "max_players": 2,
        "temporal_state_num": 39,  # 3 automata
    },
}

# Quick lookup for temporal state numbers
ENV_TEMPORAL_STATE_NUM = {
    layout: config["temporal_state_num"]
    for layout, config in ENV_CONFIGS.items()
    if config["temporal_state_num"] is not None
}


def get_env_config(layout: str) -> Dict:
    """
    Get configuration for an environment layout
    
    Args:
        layout: Layout name (e.g., "random1_m")
        
    Returns:
        Dictionary of environment configuration
    """
    if layout not in ENV_CONFIGS:
        raise ValueError(
            f"Unknown layout: {layout}. "
            f"Supported layouts: {list(ENV_CONFIGS.keys())}"
        )
    return ENV_CONFIGS[layout]


def get_env_version(layout: str) -> str:
    """Get overcooked version for a layout"""
    config = get_env_config(layout)
    return config["version"]


def get_temporal_state_num(layout: str) -> int:
    """Get temporal state number for a layout (for LTL)"""
    config = get_env_config(layout)
    temporal_num = config.get("temporal_state_num")
    if temporal_num is None:
        raise ValueError(f"Layout {layout} does not support temporal states")
    return temporal_num


def list_envs():
    """List all supported environments"""
    return {
        layout: {
            "version": config["version"],
            "temporal_state_num": config["temporal_state_num"]
        }
        for layout, config in ENV_CONFIGS.items()
    }
