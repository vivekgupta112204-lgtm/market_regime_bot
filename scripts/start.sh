#!/bin/bash
# Master boot sequence resolving daemon deployments

set -e
source venv/bin/activate

echo "Booting HMM Terminal Platform..."
# Exec is used to replace the shell process so signals route right to python
exec python run_phase12.py
