"""Top-level Manager orchestrating AI agent swarm."""

from typing import Dict, Any
from loguru import logger
import numpy as np
import pandas as pd

from ai.research_agent import ResearchAgent
from ai.news_agent import NewsAgent
from ai.portfolio_agent import PortfolioAgent
from ai.execution_agent import ExecutionAgent
from ai.risk_agent import RiskAgent

class AgentManager:
    """Supervisor Agent for all subordinate modular AI workers."""
    
    def __init__(self, config_weights: dict = None):
        if config_weights is None:
            self.weights = {
                "research": 0.40,
                "news": 0.15,
                "risk": 0.30,
                "portfolio": 0.15
            }
        else:
            self.weights = config_weights
            
        self.research = ResearchAgent()
        self.news = NewsAgent()
        self.portfolio = PortfolioAgent()
        self.execution = ExecutionAgent()
        self.risk = RiskAgent()
        
        logger.info("AI Supervisor Agent initialized with full institutional ensemble.")

    def run_consensus_engine(self, market_data: pd.DataFrame, symbols: list[str], cov_matrix: np.ndarray, expected_returns: np.ndarray, hist_returns: np.ndarray, current_drawdown: float, order_size: int, avg_volume: float, atr: float, spread: float) -> Dict[str, Any]:
        """Queries subordinate agents and produces weighted consensus execution."""
        logger.debug("Supervisor querying underlying swarm intelligence...")
        
        # 1. Research Agent
        res_data = self.research.analyze(market_data)
        regime = res_data.get("current_regime_estimate", "Unknown")
        research_score = 1.0 if regime == "Bull" else (-1.0 if regime == "Bear" else 0.0)
        
        # 2. News Agent
        news_data = self.news.deduce_sentiment([])
        news_score = news_data.get("Overall_Sentiment_Score", 0.0)
        
        # 3. Portfolio Agent
        port_data = self.portfolio.evaluate_allocation(symbols, expected_returns, cov_matrix)
        port_score = 1.0 if port_data.get("risk_ok", False) else -1.0
        
        # 4. Risk Agent
        risk_data = self.risk.check_exposure(port_data.get("suggested_allocations", {}), hist_returns, current_drawdown)
        risk_score = 1.0 if risk_data.get("approved", False) else -1.0
        
        # 5. Weighted Score
        raw_score = (
            (research_score * self.weights["research"]) +
            (news_score * self.weights["news"]) +
            (port_score * self.weights["portfolio"]) +
            (risk_score * self.weights["risk"])
        )
        
        if risk_data.get("approved", False) == False:
            final_action = "NO TRADE"
            confidence = 1.0
        elif raw_score >= 0.4:
            final_action = "BUY"
            confidence = raw_score
        elif raw_score <= -0.4:
            final_action = "SELL"
            confidence = abs(raw_score)
        elif raw_score > 0:
            final_action = "HOLD"
            confidence = raw_score
        else:
            final_action = "NO TRADE"
            confidence = abs(raw_score)
            
        # 6. Execution Agent
        if final_action in ["BUY", "SELL"]:
            exec_data = self.execution.suggest_routes(order_size, avg_volume, atr, spread, urgency_factor=confidence)
        else:
            exec_data = None
            
        decision_log = {
            "final_action": final_action,
            "confidence": float(confidence),
            "raw_score": float(raw_score),
            "inputs": {
                "research": res_data,
                "news": news_data,
                "risk": risk_data,
                "portfolio": port_data
            },
            "execution": exec_data
        }
        
        logger.info(f"Agent Consensus reached: {final_action} with {confidence*100:.1f}% confidence")
        return decision_log
