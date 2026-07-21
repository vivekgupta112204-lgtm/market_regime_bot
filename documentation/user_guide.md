# Trading User Guide

### Booting Live Mode
Simply invoke `./scripts/start.sh` on standard UNIX limits. The orchestrator will parse `configs/production.yaml` binding to active `BrokerManager` structs.

### Interacting with Dashboards
The FastAPI server spawns a live streaming process via Uvicorn. Access the Streamlit dash locally on port `8501`.
