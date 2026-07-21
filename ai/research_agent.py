"""Subordinate Agent for Quantitative Research."""

import joblib
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

class ResearchAgent:
    """Uses a trained Hidden Markov Model to deduce market regimes."""
    
    def __init__(self, model_path: str = "saved_models/hmm_best.pkl", scaler_path: str = "saved_models/scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self._model = None
        self._scaler = None
        self._regime_map = {0: "Bear", 1: "Bull", 2: "High Volatility", 3: "Sideways"}
        self._load_dependencies()

    def _load_dependencies(self):
        try:
            self._model = joblib.load(self.model_path)
            self._scaler = joblib.load(self.scaler_path)
            logger.info("ResearchAgent successfully loaded HMM and Scaler.")
        except Exception as e:
            logger.warning(f"ResearchAgent could not load models: {e}. Will fallback to safe mode.")

    def analyze(self, market_data: pd.DataFrame) -> dict:
        """Processes current market states to deduce regime shifts using HMM."""
        if self._model is None or market_data.empty:
            return {
                "current_regime_estimate": "Unknown",
                "confidence": 0.0,
                "probabilities": {},
                "transition_matrix": [],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        try:
            # Assume market_data has necessary features, run scaling
            scaled_features = self._scaler.transform(market_data)
            
            # Predict hidden state
            hidden_state = self._model.predict(scaled_features)[-1]
            state_probs = self._model.predict_proba(scaled_features)[-1]
            
            regime = self._regime_map.get(hidden_state, f"State_{hidden_state}")
            confidence = float(np.max(state_probs))
            
            return {
                "current_regime_estimate": regime,
                "confidence": confidence,
                "probabilities": {self._regime_map.get(i, str(i)): float(p) for i, p in enumerate(state_probs)},
                "transition_matrix": self._model.transmat_.tolist(),
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"HMM Analysis failed: {e}")
            return {
                "current_regime_estimate": "Error",
                "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
