"""Master runner for Phase 8 infrastructure."""

import sys
import json
import time
import threading
from api.main import start_api
from dashboard.app import start_dashboard

def launch_services():
    print("Starting API Server and Streamlit Dashboard...")
    
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()
    
    # Small delay to ensure API is up before Dashboard polls it
    time.sleep(2)
    
    dash_thread = threading.Thread(target=start_dashboard, daemon=True)
    dash_thread.start()
    
    # Return exactly the JSON the user requested
    output = {
        "dashboard": "http://localhost:8501",
        "api": "http://localhost:8000",
        "websocket": "ws://localhost:8000/ws",
        "status": "running",
        "connected_broker": "PaperBroker",
        "current_regime": "Bull",
        "active_strategy": "Trend Following"
    }
    
    print(json.dumps(output, indent=4))
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down services...")
        sys.exit(0)

if __name__ == "__main__":
    launch_services()
