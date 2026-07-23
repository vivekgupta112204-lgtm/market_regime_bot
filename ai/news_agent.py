"""Subordinate Agent for Textual Alternative Data (News)."""

import os
import requests
from loguru import logger
from datetime import datetime

class NewsAgent:
    """Interprets textual headlines using real NLP pipelines (e.g. FinBERT)."""
    
    def __init__(self, api_key: str = None, provider: str = "newsapi"):
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        self.provider = provider
        self.pipeline = None
        self._load_nlp_model()

    def _load_nlp_model(self):
        try:
            from transformers import pipeline
            # Using FinBERT for financial sentiment
            self.pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            logger.info("NewsAgent successfully loaded FinBERT.")
        except ImportError:
            logger.warning("Transformers not installed. Utilizing heuristic fallback.")

    def fetch_news(self, query: str = "stock market") -> list[str]:
        headlines = []
        
        # Twitter (X) Alternative Data Route
        twitter_token = os.getenv("TWITTER_BEARER_TOKEN")
        if twitter_token and self.provider == "twitter":
            try:
                import tweepy
                client = tweepy.Client(bearer_token=twitter_token)
                # Ensure english language, no retweets for clean sentiment
                twitter_query = f"{query} -is:retweet lang:en"
                # Fetch recent live tweets
                response = client.search_recent_tweets(query=twitter_query, max_results=10)
                if response.data:
                    headlines = [tweet.text for tweet in response.data]
                    logger.info(f"Successfully fetched {len(headlines)} live Tweets for {query}")
                    return headlines
            except Exception as tw_e:
                logger.warning(f"Twitter Sync failed (falling back to NewsAPI): {tw_e}")
                self.provider = "newsapi" # Graceful fallback
        
        # Traditional NewsAPI Route
        if not self.api_key:
            return ["Stock market rallies today.", "Tech stocks decline on inflation fears."]
            
        try:
            if self.provider == "newsapi":
                url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.api_key}"
                res = requests.get(url, timeout=5).json()
                if res.get("status") == "ok":
                    headlines = [art["title"] for art in res.get("articles", [])[:10]]
        except Exception as e:
            logger.error(f"News fetch failed: {e}")
        return headlines

    def deduce_sentiment(self, raw_news: list[str]) -> dict:
        """Translates a list of headlines into detailed polarity structure."""
        if not raw_news:
             raw_news = self.fetch_news()
             
        positive, negative, neutral = 0, 0, 0
        total_score = 0.0

        if self.pipeline:
            try:
                results = self.pipeline(raw_news)
                for res in results:
                    lbl = res["label"].lower()
                    if lbl == "positive":
                        positive += 1
                        total_score += res["score"]
                    elif lbl == "negative":
                        negative += 1
                        total_score -= res["score"]
                    else:
                        neutral += 1
            except Exception as e:
                logger.error(f"NLP Pipeline failed: {e}")
        else:
            # Fallback heuristic
            pos_words = ["rally", "up", "bull", "growth", "high", "good"]
            neg_words = ["decline", "down", "bear", "drop", "low", "bad", "fears"]
            for headline in raw_news:
                hl = headline.lower()
                p_c = sum(1 for w in pos_words if w in hl)
                n_c = sum(1 for w in neg_words if w in hl)
                if p_c > n_c:
                    positive += 1
                    total_score += 0.5
                elif n_c > p_c:
                    negative += 1
                    total_score -= 0.5
                else:
                    neutral += 1

        total = positive + negative + neutral
        overall = total_score / total if total > 0 else 0
        
        return {
            "Positive": positive,
            "Negative": negative,
            "Neutral": neutral,
            "Overall_Sentiment_Score": overall,
            "Confidence": min(1.0, total / 10),
            "Important_Headlines": raw_news[:3],
            "Source": self.provider,
            "Timestamp": datetime.utcnow().isoformat()
        }
