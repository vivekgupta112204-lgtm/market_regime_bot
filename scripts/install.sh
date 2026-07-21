#!/bin/bash
# High-Level Unix Installer deploying HMM-Bot Platform

set -e
echo "Starting Enterprise Autonomous Platform Installer..."

# Enforce Python Version constraints
python3 -c 'import sys; assert sys.version_info >= (3,10)' || { echo "Python 3.10+ required."; exit 1; }

# Initialize VENV
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run structural builder
python -c "from release.installer import Installer; inst = Installer(); inst.verify_environment(); inst.initialize_database()"

echo "System Installed successfully."
