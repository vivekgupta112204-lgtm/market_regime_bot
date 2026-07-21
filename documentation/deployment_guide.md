# Bare-metal Deployment Guide

## Prerequisites
- **Git**
- **Python 3.10+**

## Sequence
1. Give execution rights to the scripts: `chmod +x scripts/*.sh`
2. Run baseline VENV construction: `./scripts/install.sh`
3. Verify secrets vault `config/.secrets.enc` exists with decrypted ALPACA keys.
4. Issue `./scripts/start.sh`. Process will execute detached cleanly.
