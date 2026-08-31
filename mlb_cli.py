#!/usr/bin/env python3
"""
MLB CLI Entrypoint Script
Run with --help to see all available --switches.
"""

import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from mlb_engine.cli import main

if __name__ == "__main__":
    main()
