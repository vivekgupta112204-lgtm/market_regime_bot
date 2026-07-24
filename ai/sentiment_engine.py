"""
Real-Time News Sentiment Analyzer (NLP Financial Intelligence).
Uses Google Gemini to parse live financial headlines and extract market sentiment scores.
"""

import os
import yfinance as yf
from loguru import logger
from google import genai
from dotenv import load_dotenv
import re

load_dotenv()


class NewsSentimentAnalyzer:
    """Scrapes live financial headlines and converts them into numerical sentiment tensors."""
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("GEMINI_API_KEY missing. News Sentiment will return neutral.")
    
    def analyze_sentiment(self, symbol: str) -> dict:
        """
        Fetches live news headlines for a symbol and uses Gemini LLM 
        to extract a financial sentiment score (-1.0 to +1.0).
        """
        logger.info(f"📰 Scanning live news sentiment for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            news_items = ticker.news
            
            if not news_items or len(news_items) == 0:
                return {"score": 0.0, "headlines": [], "verdict": "NEUTRAL"}
            
            # Extract top 5 headlines
            headlines = []
            for item in news_items[:5]:
                title = item.get("title", "")
                if title:
                    headlines.append(title)
            
            if not headlines:
                return {"score": 0.0, "headlines": [], "verdict": "NEUTRAL"}
            
            headlines_text = "\n".join([f"- {h}" for h in headlines])
            logger.info(f"📋 Found {len(headlines)} headlines for {symbol}")
            
            if not self.client:
                logger.critical("🚨 ALERT: Sentiment API Key Missing. Engine Unavailable.")
                return {"score": 0.0, "headlines": headlines, "verdict": "UNAVAILABLE_NO_API"}
            
            # Use Gemini to analyze sentiment
            prompt = (
                f"You are a quantitative financial sentiment analyzer. "
                f"Analyze these news headlines about {symbol} and determine market sentiment.\n\n"
                f"Headlines:\n{headlines_text}\n\n"
                f"Output EXACTLY in this format:\n"
                f"SENTIMENT_SCORE: X.XX (between -1.00 for extremely bearish to +1.00 for extremely bullish)\n"
                f"VERDICT: BULLISH or BEARISH or NEUTRAL\n"
                f"SUMMARY: One sentence summary of the news mood."
            )
            
            import threading
            response = None
            
            def _call():
                nonlocal response
                response = self.client.models.generate_content(
                    model='gemini-1.5-flash',
                    contents=prompt,
                )
                
            t = threading.Thread(target=_call)
            t.start()
            t.join(timeout=8.0)
            
            if t.is_alive():
                logger.error("🚨 ALERT: Sentiment API Call Timeout.")
                return {"score": 0.0, "headlines": headlines, "verdict": "UNAVAILABLE_TIMEOUT"}
            
            if not response:
                return {"score": 0.0, "headlines": headlines, "verdict": "UNAVAILABLE_ERROR"}
                
            reply = response.text.strip()
            
            # Parse score
            score = 0.0
            score_match = re.search(r'SENTIMENT_SCORE:\s*([-+]?\d*\.?\d+)', reply, re.IGNORECASE)
            if score_match:
                score = max(-1.0, min(1.0, float(score_match.group(1))))
            
            # Parse verdict
            verdict = "NEUTRAL"
            if "BULLISH" in reply.upper():
                verdict = "BULLISH"
            elif "BEARISH" in reply.upper():
                verdict = "BEARISH"
            
            logger.info(f"🧠 NLP Sentiment for {symbol}: Score={score:+.2f} | Verdict={verdict}")
            
            return {"score": score, "headlines": headlines, "verdict": verdict}
            
        except Exception as e:
            logger.error(f"🚨 ALERT: News sentiment analysis failed for {symbol}: {e}")
            return {"score": 0.0, "headlines": [], "verdict": "UNAVAILABLE_ERROR"}


class MultiTimeframeConfluence:
    """
    Analyzes a stock across 3 timeframes (5m, 1h, 1d) simultaneously.
    Trade is only high-conviction when ALL timeframes agree on direction.
    """
    
    def analyze_confluence(self, symbol: str) -> dict:
        """
        Returns:
        - confluence_score: 0 to 3 (how many timeframes agree)
        - direction: BULLISH / BEARISH / MIXED
        - timeframe_signals: dict of individual TF results
        """
        logger.info(f"🔭 Multi-Timeframe Confluence Analysis for {symbol}...")
        
        timeframes = {
            "5m": {"period": "5d", "interval": "5m"},
            "1h": {"period": "1mo", "interval": "1h"},
            "1d": {"period": "3mo", "interval": "1d"},
        }
        
        signals = {}
        bullish_count = 0
        bearish_count = 0
        
        for tf_name, tf_params in timeframes.items():
            try:
                data = yf.download(symbol, period=tf_params["period"], interval=tf_params["interval"], progress=False)
                
                if data.empty or len(data) < 30:
                    signals[tf_name] = "INSUFFICIENT_DATA"
                    continue
                
                close = data['Close']
                
                # EMA Trend
                ema_fast = close.ewm(span=12).mean().iloc[-1]
                ema_slow = close.ewm(span=26).mean().iloc[-1]
                current = close.iloc[-1]
                
                # RSI
                delta = close.diff()
                gain = delta.where(delta > 0, 0.0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
                rs = gain / (loss + 1e-10)
                rsi = float((100 - (100 / (1 + rs))).iloc[-1])
                
                # Decision: Bullish if price > both EMAs AND RSI > 50
                if current > ema_fast and current > ema_slow and rsi > 50:
                    signals[tf_name] = "BULLISH"
                    bullish_count += 1
                elif current < ema_fast and current < ema_slow and rsi < 50:
                    signals[tf_name] = "BEARISH"
                    bearish_count += 1
                else:
                    signals[tf_name] = "NEUTRAL"
                    
            except Exception as e:
                signals[tf_name] = "ERROR"
        
        # Confluence calculation
        if bullish_count == 3:
            direction = "STRONG_BULLISH"
            confluence = 3
        elif bearish_count == 3:
            direction = "STRONG_BEARISH"
            confluence = 3
        elif bullish_count >= 2:
            direction = "LEAN_BULLISH"
            confluence = 2
        elif bearish_count >= 2:
            direction = "LEAN_BEARISH"
            confluence = 2
        else:
            direction = "MIXED"
            confluence = max(bullish_count, bearish_count)
        
        logger.info(f"🔭 Confluence Result for {symbol}: {direction} ({confluence}/3 TFs aligned)")
        logger.info(f"   5m={signals.get('5m','?')} | 1h={signals.get('1h','?')} | 1d={signals.get('1d','?')}")
        
        return {
            "confluence_score": confluence,
            "direction": direction,
            "timeframe_signals": signals
        }
