#!/bin/bash
# Reverts directories forcefully from specific zip payloads

if [ -z "$1" ]
then
    echo "Usage: ./restore.sh <backup_filename.zip>"
    exit 1
fi

echo "Wiping current un-tracked state and forcefully overriding volume directories..."
unzip -o $1
echo "Restoration absolute."
