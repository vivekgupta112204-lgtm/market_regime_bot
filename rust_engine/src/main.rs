use zeromq::{Socket, SocketRecv, SubSocket};
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

    info!("🚀 Booting Rust HFT Execution Hands (ZeroMQ + Tokio)...");

    // Spawn the OrderBook WebSocket listener in a detached parallel Tokio thread
    task::spawn(async {
        if let Err(e) = orderbook::start_l2_websocket_stream().await {
            error!("WebSocket Stream Terminated: {:?}", e);
        }
    });

    // Main thread acts as the ZeroMQ SUB Listener
    let mut socket = SubSocket::new();
    
    // Connect to Python Brain on port 5555
    let endpoint = "tcp://127.0.0.1:5555";
    info!("🔗 Connecting ZMQ SUB socket to Python Brain at {}", endpoint);
    socket.connect(endpoint).await?;
    
    // Subscribe to the "TRADE" topic
    socket.subscribe("TRADE").await?;

    info!("✅ Rust Hands Listening for signals...");
    
    loop {
        // Block asynchronously until a message comes from Python
        match socket.recv().await {
            Ok(msg) => {
                // message.get(0) usually contains the topic + payload depending on frame structure
                // With pure string sends from pyzmq: b"TRADE {\"symbol\":...}"
                if let Some(bytes) = msg.get(0) {
                    let raw_str = String::from_utf8_lossy(bytes);
                    if raw_str.starts_with("TRADE ") {
                        let json_payload = &raw_str[6..]; // Strip "TRADE " prefix
                        
                        match serde_json::from_str::<TradeSignal>(json_payload) {
                            Ok(signal) => {
                                info!("⚡ RECEIVED BRAIN SIGNAL: {:?}", signal);
                                
                                // Execute immediately
                                let _ = execution::execute_alpaca_order(
                                    &signal.symbol, 
                                    &signal.side, 
                                    signal.qty
                                ).await;
                            }
                            Err(e) => error!("Failed to deserialize signal: {}", e),
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
