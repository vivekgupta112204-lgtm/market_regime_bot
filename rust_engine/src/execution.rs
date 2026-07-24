use reqwest::Client;
use serde_json::json;
use tracing::{info, error};
use std::env;

/// Extremely fast Rust-native REST execution bypassing Python's GIL.
pub async fn execute_alpaca_order(symbol: &str, side: &str, qty: f64) -> Result<(), Box<dyn std::error::Error>> {
    let api_key = env::var("ALPACA_API_KEY").unwrap_or_default();
    let secret = env::var("ALPACA_SECRET_KEY").unwrap_or_default();
    
    if api_key.is_empty() {
        error!("ALPACA_API_KEY missing. Cannot route trade.");
        return Ok(());
    }

    let url = "https://paper-api.alpaca.markets/v2/orders";
    
    let client = Client::new();
    
    let payload = json!({
        "symbol": symbol,
        "qty": qty,
        "side": side.to_lowercase(),
        "type": "market",
        "time_in_force": "day"
    });

    info!("🚀 Executing Market {side} for {qty} {symbol} directly from Rust...");

    let res = client.post(url)
        .header("APCA-API-KEY-ID", api_key)
        .header("APCA-API-SECRET-KEY", secret)
        .json(&payload)
        .send()
        .await?;

    if res.status().is_success() {
        info!("✅ Trade filled instantly via Rust Engine!");
    } else {
        error!("❌ Trade failed: {:?}", res.text().await?);
    }

    Ok(())
}
