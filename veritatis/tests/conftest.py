"""Pytest configuration and shared test setup."""

# Ensure project root (containing the 'veritatis' package) is on sys.path for tests.
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
