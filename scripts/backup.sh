#!/bin/bash
# Generates zip archives backing up internal configs manually

timestamp=$(date +%Y%m%d_%H%M%S)
zip -r backups/shell_backup_$timestamp.zip data/ config/ saved_models/
echo "Manually archived cluster footprint to shell_backup_$timestamp.zip"
