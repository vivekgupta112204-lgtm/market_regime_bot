import zmq
import json
from loguru import logger

class ZMQDispatcher:
    """
    Python Brain Publisher (PUB).
    Dispatches ML-driven trade signals to the Rust Execution Engine over ZeroMQ.
    """
    def __init__(self, port: int = 5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.bind_url = f"tcp://127.0.0.1:{port}"
        
        try:
            self.socket.bind(self.bind_url)
            logger.info(f"🧠 Python Brain ZMQ Dispatcher bound to {self.bind_url}")
        except Exception as e:
            logger.error(f"ZMQ Bind Failed: {e}. Is port {port} already in use?")
            raise e

    def publish_trade(self, symbol: str, side: str, qty: float):
        """
        Sends an ultra-lightweight Protobuf/JSON schema to the Rust 'Hands'.
        """
        payload = {
            "symbol": symbol,
            "side": side.upper(),
            "qty": qty
        }
        
        try:
            # We prefix the topic 'TRADE' for ZMQ SUB filtering on the Rust side
            message = f"TRADE {json.dumps(payload)}"
            self.socket.send_string(message)
            logger.success(f"⚡ Brain Dispatched Signal -> Rust Hands: {payload}")
        except Exception as e:
            logger.error(f"Failed to publish ZMQ trade signal: {e}")

if __name__ == "__main__":
    # Test dispatch
    import time
    dispatcher = ZMQDispatcher()
    while True:
        dispatcher.publish_trade("SPY", "BUY", 10.0)
        time.sleep(5)
