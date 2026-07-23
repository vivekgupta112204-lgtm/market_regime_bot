"""American US Intraday Stock Scanner Module (Phase 2: Multi-Factor Quantitative Engine)."""

import time
import pandas as pd
import yfinance as yf
from loguru import logger
import numpy as np

class USIntradayScanner:
    """Scans highly liquid US Mega-caps using a 12-Factor Institutional Quantitative Alpha Pipeline."""
    
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

    def _compute_obv_divergence(self, close: pd.Series, volume: pd.Series) -> str:
        """Detects On-Balance Volume (OBV) Divergence — hidden institutional accumulation/distribution."""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        # Compare last 20 bars: if price making higher highs but OBV making lower highs = BEARISH divergence
        price_slope = float(close.iloc[-1] - close.iloc[-20]) if len(close) >= 20 else 0.0
        obv_slope = float(obv.iloc[-1] - obv.iloc[-20]) if len(obv) >= 20 else 0.0
        
        if price_slope > 0 and obv_slope < 0:
            return "BEARISH_DIV"  # Price up, volume down = smart money exiting
        elif price_slope < 0 and obv_slope > 0:
            return "BULLISH_DIV"  # Price down, volume up = smart money accumulating
        elif price_slope > 0 and obv_slope > 0:
            return "CONFIRMED_UP"  # Both agree = strong trend
        else:
            return "NEUTRAL"

    def _compute_fibonacci_zone(self, close: pd.Series) -> str:
        """Identifies current price position relative to Fibonacci Retracement levels."""
        recent_high = float(close.rolling(50).max().iloc[-1])
        recent_low = float(close.rolling(50).min().iloc[-1])
        current = float(close.iloc[-1])
        
        fib_range = recent_high - recent_low
        if fib_range < 0.01:
            return "FLAT"
        
        # Key Fibonacci levels
        fib_618 = recent_low + 0.618 * fib_range  # Golden Ratio
        fib_500 = recent_low + 0.500 * fib_range
        fib_382 = recent_low + 0.382 * fib_range
        
        if current >= fib_618:
            return "ABOVE_GOLDEN"  # Above 61.8% = strong breakout territory
        elif current >= fib_500:
            return "ABOVE_50"
        elif current >= fib_382:
            return "RETRACEMENT_ZONE"  # Dangerous retracement area
        else:
            return "DEEP_PULLBACK"

    def _compute_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Computes Average Directional Index (ADX) to measure trend STRENGTH (not direction)."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        return float(adx.iloc[-1])

    def _compute_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
        """Computes Stochastic Oscillator (%K, %D) for overbought/oversold momentum."""
        high = data['High'].rolling(k_period).max()
        low = data['Low'].rolling(k_period).min()
        close = data['Close']
        
        k_pct = 100 * (close - low) / (high - low + 1e-10)
        d_pct = k_pct.rolling(d_period).mean()
        return float(k_pct.iloc[-1]), float(d_pct.iloc[-1])

    def _compute_smart_money_flow(self, data: pd.DataFrame) -> float:
        """Computes Chaikin Money Flow (CMF) — detects institutional accumulation vs distribution."""
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        
        mfm = ((close - low) - (high - close)) / (high - low + 1e-10)  # Money Flow Multiplier
        mfv = mfm * volume  # Money Flow Volume
        
        cmf = mfv.rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)
        return float(cmf.iloc[-1])

    def scan_morning_opportunities(self) -> list[str]:
        """Runs the 12-Factor Institutional Quantitative Scan for high-conviction alpha entries."""
        logger.info("⚡ Initiating Phase-3 INSTITUTIONAL 12-Factor Quantitative Alpha Scan...")
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
                    
                    # ═══════════════════════════════════════════
                    # FACTOR 8: OBV Divergence (Smart Money Detection)
                    # ═══════════════════════════════════════════
                    obv_signal = self._compute_obv_divergence(close, vol)
                    
                    if obv_signal == "CONFIRMED_UP":
                        score += 2.0  # Price + Volume both confirming = strongest signal
                    elif obv_signal == "BULLISH_DIV":
                        score += 1.5  # Hidden accumulation
                    elif obv_signal == "BEARISH_DIV":
                        score -= 2.0  # Smart money exiting = deadly trap
                    
                    # ═══════════════════════════════════════════
                    # FACTOR 9: Fibonacci Golden Ratio Position
                    # ═══════════════════════════════════════════
                    fib_zone = self._compute_fibonacci_zone(close)
                    
                    if fib_zone == "ABOVE_GOLDEN":
                        score += 2.0  # Above 61.8% = breakout territory
                    elif fib_zone == "ABOVE_50":
                        score += 1.0
                    elif fib_zone == "DEEP_PULLBACK":
                        score -= 1.0  # Too deep = trend may be broken
                    
                    # ═══════════════════════════════════════════
                    # FACTOR 10: ADX Trend Strength
                    # ═══════════════════════════════════════════
                    adx = self._compute_adx(data)
                    
                    if adx > 40:
                        score += 2.0  # Ultra-strong trend (institutional momentum)
                    elif adx > 25:
                        score += 1.0  # Healthy trending market
                    elif adx < 15:
                        score -= 1.0  # No trend = choppy noise (scalping death zone)
                    
                    # ═══════════════════════════════════════════
                    # FACTOR 11: Stochastic Oscillator Momentum
                    # ═══════════════════════════════════════════
                    stoch_k, stoch_d = self._compute_stochastic(data)
                    
                    if stoch_k > stoch_d and stoch_k < 80:
                        score += 1.0  # Bullish crossover, not overbought
                    elif stoch_k > 80 and stoch_d > 80:
                        score -= 0.5  # Both overbought = reversal risk
                    elif stoch_k < 20:
                        score -= 0.5  # Oversold, not ideal for long momentum
                    
                    # ═══════════════════════════════════════════
                    # FACTOR 12: Chaikin Money Flow (Institutional Accumulation)
                    # ═══════════════════════════════════════════
                    cmf = self._compute_smart_money_flow(data)
                    
                    if cmf > 0.15:
                        score += 2.0  # Heavy institutional accumulation
                    elif cmf > 0.05:
                        score += 1.0  # Moderate inflow
                    elif cmf < -0.15:
                        score -= 2.0  # Heavy institutional distribution (dumping)
                    elif cmf < -0.05:
                        score -= 1.0  # Moderate outflow
                    
                    # Re-append with updated full score (overwrite previous)
                    scored_picks[-1] = (symbol, round(score, 2), round(rsi, 1), macd_signal, bb_pos, round(rvol, 2))
                    
                except Exception as e:
                    pass
        except Exception as e:
            logger.error(f"Scan strictly failed: {e}")
        
        # Sort by composite score (highest first)
        scored_picks.sort(key=lambda x: x[1], reverse=True)
        
        # Log the leaderboard
        logger.info("═══════════════════════════════════════════════════════════════")
        logger.info("  📊 12-FACTOR INSTITUTIONAL QUANTITATIVE ALPHA LEADERBOARD")
        logger.info("═══════════════════════════════════════════════════════════════")
        for rank, (sym, sc, rsi, macd, bb, rv) in enumerate(scored_picks[:20], 1):
            logger.info(f"  #{rank:02d} | {sym:6s} | Score: {sc:5.1f} | RSI: {rsi:5.1f} | MACD: {macd:15s} | BB: {bb:16s} | RVol: {rv:.1f}x")
        logger.info("═══════════════════════════════════════════════════════════════")
        
        logger.info(f"Phase-3 Scan Complete. {len(scored_picks)} assets qualified from 12-Factor Institutional pipeline.")
        
        # Return only the top 15 symbols sorted by score
        return [s[0] for s in scored_picks[:15]]
