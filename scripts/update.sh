#!/bin/bash
# Live Upgrade application handler

set -e
echo "Fetching updates from Origin Master..."
git pull origin master

echo "Applying migration states..."
source venv/bin/activate
python -c "from release.migration import MigrationHandler; mgr = MigrationHandler(); mgr.run_migrations()"

echo "Platform updated. Restart recommended."
