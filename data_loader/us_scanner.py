"""American US Intraday Stock Scanner Module (Phase 2: Multi-Factor Quantitative Engine)."""

import time
import pandas as pd
import yfinance as yf
from loguru import logger
import numpy as np

class USIntradayScanner:
    """Scans highly liquid US Mega-caps using a 7-Factor Quantitative Filter Pipeline."""
    
    def __init__(self, symbols_list: list[str] | None = None):
        if not symbols_list:
            # Expanded to 70 High-Volume companies (Tech, Finance, Crypto-proxies, Retail, Energy, Telecom, Healthcare, Banks)
            self.symbols = [
                "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD", "NFLX", "QQQ",
                "PLTR", "SOFI", "COIN", "INTC", "ARM", "BAC", "JPM", "DIS", "WMT", "SPY",
                "UBER", "HOOD", "MARA", "RIOT", "CRM", "XOM", "CVX", "UNH", "JNJ", "PFE",
                "MRK", "PEP", "KO", "MCD", "NKE", "SBUX", "BA", "CAT", "V", "MA",
                "PYPL", "SQ", "CRWD", "SNOW", "BABA", "MSTR", "GME", "AMC", "SMCI", "AVGO",
                "CSCO", "ORCL", "IBM", "ABT", "TMO", "COST", "HD", "PG", "TMUS", "VZ",
                "WFC", "C", "GS", "MS", "BLK", "ISRG", "SYK", "MDT", "ZTS", "BAX"
            ]
        else:
            self.symbols = symbols_list

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> float:
        """Computes Relative Strength Index (RSI) for overbought/oversold detection."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def _compute_macd_signal(self, series: pd.Series) -> str:
        """Computes MACD crossover signal (BULLISH / BEARISH / NEUTRAL)."""
        ema12 = series.ewm(span=12).mean()
        ema26 = series.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        
        if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            return "BULLISH_CROSS"
        elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            return "BEARISH_CROSS"
        elif macd_line.iloc[-1] > signal_line.iloc[-1]:
            return "BULLISH"
        else:
            return "BEARISH"
    
    def _compute_bollinger_position(self, series: pd.Series, window: int = 20) -> str:
        """Checks if price is near Bollinger Band boundaries (squeeze/breakout detection)."""
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        current = series.iloc[-1]
        
        if current >= upper.iloc[-1]:
            return "UPPER_BREAKOUT"
        elif current <= lower.iloc[-1]:
            return "LOWER_BREAKOUT"
        else:
            band_width = float((upper.iloc[-1] - lower.iloc[-1]) / (sma.iloc[-1] + 1e-10))
            if band_width < 0.02:
                return "SQUEEZE"
            return "MID_RANGE"

    def _compute_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Computes Average True Range to measure intraday volatility magnitude."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return float(atr.iloc[-1])

    def scan_morning_opportunities(self) -> list[str]:
        """Runs the 7-Factor Quantitative Scan for high-conviction momentum entries."""
        logger.info("⚡ Initiating Phase-2 Multi-Factor US Quantitative Scan...")
        scored_picks = []
        
        try:
            # Downloading 5 minute data for the last 5 days
            df = yf.download(self.symbols, period="5d", interval="5m", group_by="ticker", progress=False)
            
            for symbol in self.symbols:
                try:
                    if len(self.symbols) == 1:
                        data = df
                    else:
                        data = df[symbol]
                        
                    if data.empty: continue
                    
                    close = data['Close']
                    vol = data['Volume']
                    
                    if len(close) < 50: continue
                    
                    # ═══════════════════════════════════════════
                    # FACTOR 1: EMA Trend Alignment (20 & 50)
                    # ═══════════════════════════════════════════
                    ema20 = close.ewm(span=20).mean().iloc[-1]
                    ema50 = close.ewm(span=50).mean().iloc[-1]
                    current_price = close.iloc[-1]
                    
                    if not (current_price > ema20 and current_price > ema50):
                        continue  # Hard filter: must be above both EMAs
                    
                    score = 0.0
                    
                    # Golden Cross bonus: EMA20 > EMA50 (short-term momentum > long-term)
                    if ema20 > ema50:
                        score += 1.5
                    
                    # ═══════════════════════════════════════════
                    # FACTOR 2: VWAP Dominance
                    # ═══════════════════════════════════════════
                    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
                    vwap = (typical_price * vol).cumsum() / vol.cumsum()
                    current_vwap = vwap.iloc[-1]
                    
                    if current_price > current_vwap:
                        score += 1.0
                    else:
                        continue  # Hard filter: must be above VWAP
                    
                    # ═══════════════════════════════════════════
                    # FACTOR 3: Relative Volume Surge (RVol)
                    # ═══════════════════════════════════════════
                    current_vol = vol.iloc[-1]
                    avg_vol = vol.rolling(20).mean().iloc[-1]
                    rvol = current_vol / (avg_vol + 1.0)
                    
                    if rvol < 1.2:
                        continue  # Hard filter: Minimum 1.2x volume
                    
                    # Tiered RVol scoring
                    if rvol > 3.0:
                        score += 3.0  # Extreme volume (institutional sweep)
                    elif rvol > 2.0:
                        score += 2.0  # High volume
                    else:
                        score += 1.0  # Moderate volume
                    
                    # ═══════════════════════════════════════════
                    # FACTOR 4: RSI Momentum Zone
                    # ═══════════════════════════════════════════
                    rsi = self._compute_rsi(close)
                    
                    if 50 < rsi < 70:
                        score += 1.5  # Sweet spot: strong momentum, not overbought
                    elif 40 < rsi <= 50:
                        score += 0.5  # Neutral zone
                    elif rsi >= 70:
                        score -= 1.0  # Overbought penalty (reversal risk)
                    elif rsi <= 30:
                        score -= 0.5  # Oversold (not suitable for momentum long)
                    
                    # ═══════════════════════════════════════════
                    # FACTOR 5: MACD Trend Confirmation
                    # ═══════════════════════════════════════════
                    macd_signal = self._compute_macd_signal(close)
                    
                    if macd_signal == "BULLISH_CROSS":
                        score += 2.0  # Fresh crossover = highest conviction
                    elif macd_signal == "BULLISH":
                        score += 1.0
                    elif macd_signal == "BEARISH_CROSS":
                        score -= 1.5  # Actively crossing down
                    
                    # ═══════════════════════════════════════════
                    # FACTOR 6: Bollinger Band Position
                    # ═══════════════════════════════════════════
                    bb_pos = self._compute_bollinger_position(close)
                    
                    if bb_pos == "UPPER_BREAKOUT":
                        score += 1.5  # Breakout momentum
                    elif bb_pos == "SQUEEZE":
                        score += 2.0  # Squeeze = Explosive move imminent
                    elif bb_pos == "LOWER_BREAKOUT":
                        score -= 1.0  # Crashing through floor
                    
                    # ═══════════════════════════════════════════
                    # FACTOR 7: ATR Volatility Premium
                    # ═══════════════════════════════════════════
                    atr = self._compute_atr(data)
                    atr_pct = atr / (float(current_price) + 1e-10) * 100
                    
                    if atr_pct > 1.5:
                        score += 1.0  # High ATR = bigger profit potential per scalp
                    elif atr_pct > 0.8:
                        score += 0.5
                    
                    # Append with composite score
                    scored_picks.append((symbol, round(score, 2), round(rsi, 1), macd_signal, bb_pos, round(rvol, 2)))
                    
                except Exception as e:
                    pass
        except Exception as e:
            logger.error(f"Scan strictly failed: {e}")
        
        # Sort by composite score (highest first)
        scored_picks.sort(key=lambda x: x[1], reverse=True)
        
        # Log the leaderboard
        logger.info("═══════════════════════════════════════════")
        logger.info("  📊 MULTI-FACTOR QUANTITATIVE LEADERBOARD")
        logger.info("═══════════════════════════════════════════")
        for rank, (sym, sc, rsi, macd, bb, rv) in enumerate(scored_picks[:20], 1):
            logger.info(f"  #{rank:02d} | {sym:6s} | Score: {sc:5.1f} | RSI: {rsi:5.1f} | MACD: {macd:15s} | BB: {bb:16s} | RVol: {rv:.1f}x")
        logger.info("═══════════════════════════════════════════")
        
        logger.info(f"Phase-2 Scan Complete. {len(scored_picks)} assets qualified from 7-Factor pipeline.")
        
        # Return only the top 15 symbols sorted by score
        return [s[0] for s in scored_picks[:15]]
