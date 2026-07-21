#!/bin/bash
# Graceful UNIX POSIX Shutdown commands

echo "Attempting graceful bot termination..."
pkill -f 'python run_phase12.py'
echo "Platform execution halted."
