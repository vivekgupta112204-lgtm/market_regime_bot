from dotenv import load_dotenv
load_dotenv()
"""
Multi-Agent Swarm Intelligence Engine (Phase 3: 5-Agent Adversarial Council).
Implements adversarial AI debate with weighted confidence voting and historical memory.
"""

import os
import json
import threading
import concurrent.futures
from datetime import datetime
from loguru import logger
from google import genai
from pydantic import BaseModel

class AgentResponse(BaseModel):
    reasoning: str
    score: int  # 0-100 for confidence, 1-10 for risk

class SwarmDebateEngine:
    """
    Manages a deterministic 4-Agent Advisory Council:
    1. BULL (Aggressive Buyer)
    2. BEAR (Aggressive Short Seller)
    3. RISK ANALYST (Capital Preservation Specialist)
    4. CONTRARIAN (Devil's Advocate - opposes majority)
    NOTE: The 'Supreme Judge' is strictly deterministic, non-LLM Python logic enforcing risk margins.
    """
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
             logger.critical("Swarm Engine failure: GEMINI_API_KEY not found in environment.")
             raise ValueError("GEMINI_API_KEY is required for the LLM Debate Engine.")
        self.client = genai.Client(api_key=self.api_key)
        self.debate_history = self._load_history()
        
    def _load_history(self) -> list:
        try:
            if os.path.exists("logs/swarm_logs.json"):
                with open("logs/swarm_logs.json", "r", encoding="utf-8") as f:
                    logs = [json.loads(line) for line in f.readlines() if line.strip()]
                return logs[-5:]
        except Exception:
            pass
        return []
    
    def _build_history_context(self) -> str:
        if not self.debate_history:
            return "No prior debate history available."
        
        context_lines = [
            f"- {h.get('target','?')} ({h.get('signal','?')}): Verdict was {h.get('judge','?')}" 
            for h in self.debate_history
        ]
        return "Recent trade debate outcomes:\n" + "\n".join(context_lines)

    def _query_json(self, prompt: str, is_risk: bool = False) -> dict:
        """Fetches strict JSON completions with timeout. Fails securely to veto defaults."""
        result_dict = {"reasoning": "TIMEOUT/ERROR FALLBACK", "score": 10 if is_risk else 0}
        try:
             response = None
             
             def _call():
                 nonlocal response
                 response = self.client.models.generate_content(
                     model='gemini-1.5-flash',
                     contents=prompt,
                     config=genai.types.GenerateContentConfig(
                         response_mime_type="application/json",
                         response_schema=AgentResponse,
                     ),
                 )
                 
             t = threading.Thread(target=_call)
             t.start()
             t.join(timeout=8.0)
             
             if t.is_alive():
                 logger.error("LLM API Call Timeout Exceeded (8s). Falling back to safe veto.")
                 return result_dict
                 
             if response and response.text:
                 data = json.loads(response.text)
                 result_dict["reasoning"] = data.get("reasoning", "Parse missing")
                 # Clamp scores securely
                 parsed_score = int(data.get("score", 10 if is_risk else 0))
                 result_dict["score"] = max(1, min(parsed_score, 10)) if is_risk else max(0, min(parsed_score, 100))
                 return result_dict
                 
        except Exception as e:
             logger.error(f"LLM Structure Extraction Failed: {e}")
             
        return result_dict

    def conduct_debate(self, target: str, ppo_signal: str) -> str:
        logger.info(f"⚖️ SWARM COUNCIL ACTIVATED: Deterministic Advisory Debate for {ppo_signal} on {target}")
        history_ctx = self._build_history_context()
        
        bull_prompt = f"Analyze as momentum trader. Signal: {ppo_signal} {target}. {history_ctx}. Return reasoning and confidence score (0-100)."
        bear_prompt = f"Analyze as short seller. Signal: {ppo_signal} {target}. {history_ctx}. Return reasoning and confidence score (0-100)."
        risk_prompt = f"Analyze as Chief Risk Officer. Signal: {ppo_signal} {target}. {history_ctx}. Return reasoning and risk score (1-10)."
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_bull = executor.submit(self._query_json, bull_prompt, False)
            future_bear = executor.submit(self._query_json, bear_prompt, False)
            future_risk = executor.submit(self._query_json, risk_prompt, True)
            
            bull_resp = future_bull.result()
            bear_resp = future_bear.result()
            risk_resp = future_risk.result()

        bull_conf = bull_resp["score"]
        bear_conf = bear_resp["score"]
        risk_score = risk_resp["score"]
        
        logger.info(f"📈 [BULL] ({bull_conf}%): {bull_resp['reasoning'][:80]}...")
        logger.info(f"📉 [BEAR] ({bear_conf}%): {bear_resp['reasoning'][:80]}...")
        logger.info(f"🛡️ [RISK] ({risk_score}/10): {risk_resp['reasoning'][:80]}...")

        # Contrarian Execution
        majority_is_bullish = bull_conf > bear_conf
        contrarian_stance = "against" if majority_is_bullish else "for"
        contra_prompt = f"Contrarian manager. Majority is {'bullish' if majority_is_bullish else 'bearish'} on {target}. Argue {contrarian_stance} the {ppo_signal}. Return reasoning and conviction score (0-100)."
        contra_resp = self._query_json(contra_prompt, False)
        contrarian_conf = contra_resp["score"]

        # ═══════════════════════════════════════════
        # DETERMINISTIC VERDICT (Python-based Logic)
        # ═══════════════════════════════════════════
        if risk_score >= 7:
            verdict = "[VETO]"
            logger.error(f"⛔ [SUPREME PYTHON JUDGE]: RISK OVERRIDE — Risk score {risk_score}/10 blocks trade. [VETO]")
        elif ppo_signal == "LONG" and bear_conf + contrarian_conf > bull_conf * 1.5:
            verdict = "[VETO]"
            logger.error(f"⛔ [SUPREME PYTHON JUDGE]: Opposing forces overpower Bull confidence. [VETO]")
        elif ppo_signal == "SHORT" and bull_conf + contrarian_conf > bear_conf * 1.5:
            verdict = "[VETO]"
            logger.error(f"⛔ [SUPREME PYTHON JUDGE]: Opposing forces overpower Bear confidence. [VETO]")
        elif bull_conf > 50 or bear_conf > 50:
            verdict = "[APPROVED]"
            logger.success(f"👨‍⚖️ [SUPREME PYTHON JUDGE]: Deterministic Council vote passes. [APPROVED]")
        else:
            verdict = "[VETO]"
            logger.error(f"⛔ [SUPREME PYTHON JUDGE]: Baseline conviction too low. [VETO]")
        
        try:
             log_data = {
                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "target": target,
                 "signal": ppo_signal,
                 "bull": bull_resp["reasoning"],
                 "bull_confidence": bull_conf,
                 "bear": bear_resp["reasoning"],
                 "bear_confidence": bear_conf,
                 "risk_analyst": risk_resp["reasoning"],
                 "risk_score": risk_score,
                 "contrarian": contra_resp["reasoning"],
                 "judge": verdict
             }
             os.makedirs("logs", exist_ok=True)
             with open("logs/swarm_logs.json", "a", encoding="utf-8") as f:
                 f.write(json.dumps(log_data) + "\n")
        except Exception:
             pass
             
        return verdict
