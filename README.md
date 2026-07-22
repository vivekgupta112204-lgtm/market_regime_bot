# US Serverless Trading Bot (GitHub Actions Architecture)

This repository contains a fully autonomous quantitative trading system configured to run **Serverless on GitHub Actions**. By utilizing GitHub's free runner minutes, we completely sidestep the cost of hosting a 24/7 cloud server.

## 🚀 Serverless CI/CD Architecture

- **Stateless Execution:** The bot does not run in a continuous loop. Instead, GitHub Actions spins up a fresh Ubuntu container every 5 minutes.
- **US Timezone Gatekeeper:** Because GitHub's scheduling runs strictly on UTC, we mapped a timezone-aware script (`market_time.py`) using Python's `zoneinfo` (`America/New_York`). The script automatically adjusts for US Daylight Saving Time (DST). If the market is closed, it forces the workflow to gracefully exit holding runner time minimal.
- **Automated Fallbacks:** 
    - Auto-retries `pip install` failures if caching breaks.
    - If the trading engine crashes, an autonomous `curl` webhook shoots a Telegram message tracing the failure path.
- **Auditable Log Artifacts:** At the end of every 5-minute cycle, the `run_logs.txt` execution output is zipped and attached securely to the GitHub Action run.

## 🛠 Required Secrets Configuration

For the workflow to interact with real brokers without exposing credentials, you **Must** inject the following keys into your repository settings via `Settings > Secrets and variables > Actions > New repository secret`:

1. `ALPACA_API_KEY`
2. `ALPACA_SECRET_KEY`
3. `ALPHA_VANTAGE_KEY`
4. `TELEGRAM_TOKEN` (For Crash Alerts)
5. `TELEGRAM_CHAT_ID`

## 🖥 Local Testing

Verify the US strict-time module locally:
```bash
python market_time.py
```
*(Will print "Market Open" and return Exit 0, or "Market Closed" and return Exit 1)*

Verify the serverless trace loop:
```bash
python run_bot.py
```
