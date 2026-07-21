import sys
import json
import logging
from loguru import logger
import pandas as pd
from datetime import datetime
from backtesting.backtest_engine import run_backtest

# Disable logging to cleanly output json
logger.remove()
logger.add(lambda msg: None)

def main():
    # Use dummy synthetic data since we don't have an active exchange connection
    # It mimics a 1000-candle timeseries
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
    import numpy as np
    np.random.seed(42)
    closes = np.cumsum(np.random.randn(1000)) + 100
    df = pd.DataFrame({
        'timestamp': dates,
        'open': closes + np.random.randn(1000) * 0.5,
        'high': closes + np.abs(np.random.randn(1000)),
        'low': closes - np.abs(np.random.randn(1000)),
        'close': closes,
        'volume': np.random.randint(100, 10000, size=1000)
    })
    
    try:
        results = run_backtest(df=df)
        
        output = {
            "initial_capital": results.get("initial_capital", 100000),
            "final_capital": results.get("final_capital", 100000),
            "total_return": results.get("total_return", 0),
            "annual_return": results.get("annual_return", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "sortino_ratio": results.get("sortino_ratio", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "profit_factor": results.get("profit_factor", 0),
            "win_rate": results.get("win_rate", 0),
            "total_trades": results.get("total_trades", 0),
            "average_trade": results.get("average_trade", 0),
            "best_trade": results.get("best_trade", 0),
            "worst_trade": results.get("worst_trade", 0),
            "report_path": "./reports/backtest_report.pdf"
        }
        
        print(json.dumps(output, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main()
