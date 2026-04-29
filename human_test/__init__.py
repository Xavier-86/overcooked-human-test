"""
Overcooked Human-Test: Human-AI interaction testing tool.

Play Overcooked with trained AI agents via Pygame (full mode)
or a lightweight terminal demo (standalone mode).
"""

__version__ = "0.2.0"
__author__ = "ZSC-Eval Team"

# Only import lightweight modules here.
# Heavy dependencies (torch, yaml, zsceval) are loaded lazily by cli.py
# when the user actually runs policy mode.
from .core import HumanTest
from .keyboard import KeyboardHandler
from .renderer import CLIRenderer, ASCIIRenderer

__all__ = [
    "HumanTest",
    "KeyboardHandler",
    "CLIRenderer",
    "ASCIIRenderer",
]
