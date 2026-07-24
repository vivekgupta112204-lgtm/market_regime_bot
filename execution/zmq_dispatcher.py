import zmq
import json
from loguru import logger

class ZMQDispatcher:
    """
    Python Brain Requester (REQ).
    Strictly dispatches trade signals to Rust Execution Engine using REQ/REP pattern 
    with Acknowledgments (ACKs) to prevent any microsecond packet drop.
    """
    def __init__(self, port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        # Enforce 2-second timeout to prevent Python from freezing if Rust goes down
        self.socket.setsockopt(zmq.RCVTIMEO, 2000) 
        self.socket.setsockopt(zmq.LINGER, 0)
        
        self.endpoint = f"tcp://127.0.0.1:{port}"
        
        try:
            self.socket.connect(self.endpoint)
            logger.info(f"🧠 Python Brain ZMQ [REQ] Connected to {self.endpoint}")
        except Exception as e:
            logger.error(f"ZMQ Connect Failed: {e}")
            raise e

    def publish_trade(self, symbol: str, side: str, qty: float):
        """
        Sends an ultra-lightweight JSON payload and waits for Rust ACK.
        """
        payload = {
            "symbol": symbol,
            "side": side.upper(),
            "qty": qty
        }
        
        try:
            # We enforce REQ/REP packet delivery
            message = json.dumps(payload)
            self.socket.send_string(message)
            logger.success(f"⚡ Brain Sent Strict Signal -> Rust Hands: {payload}")
            
            # Wait for strict acknowledgment from Rust
            reply = self.socket.recv_string()
            logger.info(f"🛡️ RUST ACK RECEIVED: {reply}")
            return True
            
        except zmq.error.Again:
            logger.error(f"❌ CRITICAL: Rust Execution Engine did not ACK in 2000ms! Packet unconfirmed.")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to dispatch ZMQ trade signal: {e}")
            return False

if __name__ == "__main__":
    # Test dispatch
    import time
    dispatcher = ZMQDispatcher()
    while True:
        dispatcher.publish_trade("SPY", "BUY", 10.0)
        time.sleep(5)
