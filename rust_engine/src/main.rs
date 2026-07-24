use zeromq::{Socket, SocketRecv, SocketSend, RepSocket, ZmqMessage};
use std::env;
use dotenv::dotenv;
use tracing::{info, error, Level};
use serde::{Deserialize, Serialize};
use tokio::task;

mod orderbook;
mod execution;

#[derive(Deserialize, Serialize, Debug)]
struct TradeSignal {
    symbol: String,
    side: String,
    qty: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();
    tracing_subscriber::fmt().with_max_level(Level::INFO).init();

    info!("🚀 Booting Rust HFT Execution Hands (ZeroMQ REQ/REP + Tokio)...");

    // Spawn the OrderBook WebSocket listener in a detached parallel Tokio thread
    task::spawn(async {
        if let Err(e) = orderbook::start_l2_websocket_stream().await {
            error!("WebSocket Stream Terminated: {:?}", e);
        }
    });

    // Main thread acts as the ZeroMQ REP Server Listener
    let mut socket = RepSocket::new();
    
    // Bind robustly on port 5555
    let endpoint = "tcp://127.0.0.1:5555";
    info!("🔗 Binding ZMQ REP socket on {}", endpoint);
    socket.bind(endpoint).await?;

    info!("✅ Rust Hands Listening for strict signal dispatches...");
    
    loop {
        // Block asynchronously until a message comes from Python
        match socket.recv().await {
            Ok(msg) => {
                if let Some(bytes) = msg.get(0) {
                    let json_payload = String::from_utf8_lossy(bytes);
                    
                    match serde_json::from_str::<TradeSignal>(&json_payload) {
                        Ok(signal) => {
                            info!("⚡ RECEIVED STRICT BRAIN SIGNAL: {:?}", signal);
                            
                            // Execute immediately
                            match execution::execute_alpaca_order(&signal.symbol, &signal.side, signal.qty).await {
                                Ok(_) =>  {
                                    // Send ACK back to unblock Python Brain
                                    let _ = socket.send(ZmqMessage::from(format!("ACK_SUCCESS_FILLED: {}", signal.symbol))).await;
                                },
                                Err(e) => {
                                    error!("Execution Failed: {:?}", e);
                                    let _ = socket.send(ZmqMessage::from(format!("ACK_FAIL: {:?}", e))).await;
                                }
                            }
                        }
                        Err(e) => {
                            error!("Failed to deserialize signal: {}", e);
                            let _ = socket.send(ZmqMessage::from("ACK_ERROR_INVALID_JSON")).await;
                        }
                    }
                }
            },
            Err(e) => {
                error!("ZMQ Recv Error: {:?}", e);
            }
        }
    }
}
