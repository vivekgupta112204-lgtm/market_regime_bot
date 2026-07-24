#!/bin/bash
set -e

echo "Starting AWS Rust Deployment Protocol..."

# Ensure we have Swap Space to prevent Cargo Out-Of-Memory (OOM) crush on t3.micro
if ! grep -q "swapfile" /etc/fstab; then
    echo "Allocating 2GB Swap Space..."
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab
fi

# Install Rust toolchain non-interactive if not exists
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Pull architecture changes
cd /opt/trading_bot
git pull origin main

# Load environment logic
source /opt/trading_bot/venv/bin/activate
pip install pyzmq

# Compile Rust Hands with limited job footprint to survive 1GB RAM
cd rust_engine
export PATH="$HOME/.cargo/bin:$PATH"
echo "Compiling Rust HFT Engine..."
cargo build --release -j 1

# Configure SystemD Service for Rust Engine
cat << 'EOF' > /etc/systemd/system/rust_hft.service
[Unit]
Description=Rust Ultra-Low Latency Execution Hands (ZMQ WSS)
After=network.target

[Service]
User=root
WorkingDirectory=/opt/trading_bot/rust_engine
ExecStart=/opt/trading_bot/rust_engine/target/release/hft_rust_engine
Restart=always
RestartSec=5
Environment=PATH=/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable rust_hft.service
systemctl restart rust_hft.service

echo "AWS Rust Deployment Completed Successfully!"
