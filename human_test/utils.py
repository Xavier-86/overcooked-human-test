#!/usr/bin/env python3
"""
Utility functions for human_test.
"""

import os
from typing import List, Optional


def _get_policy_pool_roots() -> List[str]:
    """Return candidate policy pool directories to search."""
    candidates = []
    if os.environ.get("POLICY_POOL"):
        candidates.append(os.environ.get("POLICY_POOL"))
    # Check next to the package root
    pkg_root = os.path.join(os.path.dirname(__file__), "..")
    candidates.append(os.path.join(pkg_root, "exp_configs", "policy_pool"))
    # Check current working directory
    candidates.append(os.path.join(os.getcwd(), "exp_configs", "policy_pool"))
    return candidates


def get_available_envs() -> List[str]:
    """Get all supported environments / layouts."""
    defaults = ["random1_m", "random3_m"]
    envs = set(defaults)

    for root in _get_policy_pool_roots():
        if os.path.isdir(root):
            for item in os.listdir(root):
                item_path = os.path.join(root, item)
                if os.path.isdir(item_path) and not item.startswith("."):
                    envs.add(item)

    return sorted(envs)


def get_available_algos() -> List[str]:
    """Get all supported algorithms."""
    return [
        "bach",
        "fcp",
        "mep",
        "trajedi",
        "hsp",
        "cole",
        "e3t",
        "sp",
    ]


def get_policy_path(env_name: str, algo: str) -> str:
    """
    Get policy file path for a given environment and algorithm.

    Args:
        env_name: Environment name
        algo: Algorithm name

    Returns:
        Policy file path (may not exist)
    """
    for root in _get_policy_pool_roots():
        possible_paths = [
            os.path.join(root, env_name, algo, "s2", "policy.pt"),
            os.path.join(root, env_name, algo, "actor_checkpoint.pt"),
            os.path.join(root, env_name, algo, "policy_config.pkl"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path

    # Fallback: return the first candidate path (will likely fail later)
    fallback_root = _get_policy_pool_roots()[0]
    return os.path.join(fallback_root, env_name, algo, "s2", "policy.pt")


def load_policy_config(policy_path: str) -> Optional[dict]:
    """
    Load policy configuration from a pickle file.

    Args:
        policy_path: Path to the policy config file

    Returns:
        Configuration dict or None
    """
    try:
        import pickle

        with open(policy_path, "rb") as f:
            config = pickle.load(f)
        return config
    except Exception as e:
        print(f"Failed to load policy config: {e}")
        return None


def check_environment() -> bool:
    """
    Check whether the environment is correctly configured.

    Returns:
        True if the basic environment is usable
    """
    checks = {
        "zsceval": False,
        "policy_pool": False,
        "pynput": False,
    }

    try:
        import zsceval

        checks["zsceval"] = True
    except ImportError:
        pass

    for root in _get_policy_pool_roots():
        if os.path.exists(root):
            checks["policy_pool"] = True
            break

    try:
        import pynput

        checks["pynput"] = True
    except ImportError:
        pass

    print("Environment check:")
    for name, status in checks.items():
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {name}")

    if not checks["zsceval"]:
        print("\n  ℹ️  Tip: ZSC-Eval not detected. Running in demo mode.")
        print("      For full functionality, install ZSC-Eval dependencies.\n")

    return checks["pynput"]


def print_banner():
    """Print welcome banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║         Human vs AI Testing Tool for Overcooked               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)
