Write-Host "Installing Rust for Windows..." -ForegroundColor Cyan

if (Get-Command cargo -ErrorAction SilentlyContinue) {
    Write-Host "Rust is already installed!" -ForegroundColor Green
}
else {
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile "rustup-init.exe"
    Write-Host "Running Rust installer quietly..."
    Start-Process -FilePath "rustup-init.exe" -ArgumentList "-y" -Wait
    Remove-Item "rustup-init.exe"
    
    # Reload environment vars in current session
    $env:Path += ";$env:USERPROFILE\.cargo\bin"
    Write-Host "Rust toolchain downloaded!" -ForegroundColor Green
}

Write-Host "Installing Python PyZMQ..." -ForegroundColor Cyan
pip install pyzmq

Write-Host "Building Rust HFT Engine..." -ForegroundColor Cyan
Set-Location "rust_engine"
cargo build --release

Write-Host "Booting ZeroMQ Engine and Python Brain..." -ForegroundColor Cyan
# Launch rust background
Start-Process -NoNewWindow -FilePath "target\release\hft_rust_engine.exe"
Write-Host "Rust Engine Hands active! Launching ZMQ Brain..." -ForegroundColor Green
Set-Location ".."
python run_bot.py
