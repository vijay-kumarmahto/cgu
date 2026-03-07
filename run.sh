#!/bin/bash

# Quick run script - activates venv and launches app
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

source .venv/bin/activate
python start.py
