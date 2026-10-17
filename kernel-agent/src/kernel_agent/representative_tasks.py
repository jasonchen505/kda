"""Dependency-free task metadata shared by controller and GPU-only environment."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def specs():
    return json.loads((ROOT/'data/representative-tasks.json').read_text())
