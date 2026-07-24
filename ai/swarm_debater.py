from dotenv import load_dotenv
load_dotenv()
"""
Multi-Agent Swarm Intelligence Engine (Phase 3: 5-Agent Adversarial Council).
Implements adversarial AI debate with weighted confidence voting and historical memory.
"""

import os
import json
import re
from datetime import datetime
from loguru import logger
from google import genai


class SwarmDebateEngine:
    """
    Manages a 5-Agent Adversarial Council:
    1. BULL (Aggressive Buyer)
    2. BEAR (Aggressive Short Seller)
    3. RISK ANALYST (Capital Preservation Specialist)
    4. CONTRARIAN (Devil's Advocate - opposes majority)
    5. SUPREME JUDGE (Final Weighted Verdict)
    """
    
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
             logger.critical("Swarm Engine failure: GEMINI_API_KEY not found in environment.")
             raise ValueError("GEMINI_API_KEY is required for the LLM Debate Engine.")
        self.client = genai.Client(api_key=self.api_key)
        self.debate_history = self._load_history()
        
    def _load_history(self) -> list:
        """Load past debate outcomes to give Agents historical context."""
        try:
            if os.path.exists("logs/swarm_logs.json"):
                with open("logs/swarm_logs.json", "r", encoding="utf-8") as f:
                    logs = [json.loads(line) for line in f.readlines() if line.strip()]
                return logs[-5:]  # Last 5 debates as memory
        except Exception:
            pass
        return []
    
    def _build_history_context(self) -> str:
        """Converts past debates into a text block for agent memory injection."""
        if not self.debate_history:
            return "No prior debate history available."
        
        context_lines = []
        for h in self.debate_history:
            context_lines.append(f"- {h.get('target','?')} ({h.get('signal','?')}): Verdict was {h.get('judge','?')}")
        return "Recent trade debate outcomes:\n" + "\n".join(context_lines)

    def conduct_debate(self, target: str, ppo_signal: str) -> str:
        """
        Conducts a 5-Agent adversarial debate with confidence-weighted voting.
        Executes concurrently to prevent blocking the main intraday tick cycle.
        """
        logger.info(f"⚖️ SWARM COUNCIL ACTIVATED: 5-Agent Adversarial Debate opened for {ppo_signal} on {target}")
        history_ctx = self._build_history_context()
        
        bull_prompt = f"You are an aggressive Wall Street momentum trader... PPO signals {ppo_signal} on {target}. {history_ctx} 2 sentences. End with 'CONFIDENCE: X%' where X is 0-100."
        bear_prompt = f"You are a legendary short seller... PPO signals {ppo_signal} on {target}. {history_ctx} 2 sentences. End with 'CONFIDENCE: X%' where X is 0-100."
        
        import concurrent.futures
        
        import random
        # Anti-Overfitting: Agent Dropout Mechanism (20% chance to drop 1-2 non-Risk agents)
        dropout_active = random.random() < 0.20
        agents_to_drop = []
        if dropout_active:
            agents_to_drop = random.sample(["BULL", "BEAR", "CONTRARIAN"], random.randint(1, 2))
            logger.info(f"🌀 AGENT DROPOUT ACTIVE: Masking {agents_to_drop} to force Judge generalization.")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            risk_prompt_init = f"You are Chief Risk Officer. A {ppo_signal} trade on {target} is proposed. {history_ctx} Analyze risk. End with 'RISK_SCORE: X/10'."
            future_risk = executor.submit(self._query_llm, risk_prompt_init)
            
            future_bull = executor.submit(self._query_llm, bull_prompt) if "BULL" not in agents_to_drop else None
            future_bear = executor.submit(self._query_llm, bear_prompt) if "BEAR" not in agents_to_drop else None
            
            risk_reply = future_risk.result()
            bull_reply = future_bull.result() if future_bull else "[OFFLINE DUE TO DROPOUT - MAKE DECISION WITHOUT BULL]"
            bear_reply = future_bear.result() if future_bear else "[OFFLINE DUE TO DROPOUT - MAKE DECISION WITHOUT BEAR]"

        bull_conf = self._extract_confidence(bull_reply) if "BULL" not in agents_to_drop else 0
        bear_conf = self._extract_confidence(bear_reply) if "BEAR" not in agents_to_drop else 0
        risk_score = self._extract_risk_score(risk_reply)
        
        if "BULL" not in agents_to_drop: logger.info(f"📈 [AGENT BULL] (Confidence: {bull_conf}%): '{bull_reply[:100]}...'")
        if "BEAR" not in agents_to_drop: logger.info(f"📉 [AGENT BEAR] (Confidence: {bear_conf}%): '{bear_reply[:100]}...'")
        logger.info(f"🛡️ [AGENT RISK] (Risk: {risk_score}/10): '{risk_reply[:100]}...'")
        
        if "CONTRARIAN" not in agents_to_drop:
            majority_is_bullish = bull_conf > bear_conf
            contrarian_stance = "against" if majority_is_bullish else "for"
            contrarian_prompt = (
                f"You are a contrarian manager. Majority is {'bullish' if majority_is_bullish else 'bearish'} on {target}. "
                f"Argue {contrarian_stance} the {ppo_signal} trade in 1 sentence. Then write 'CONVICTION: X%'."
            )
            contrarian_reply = self._query_llm(contrarian_prompt)
            contrarian_conf = self._extract_confidence(contrarian_reply, keyword="CONVICTION")
        else:
            contrarian_reply = "[OFFLINE DUE TO DROPOUT - MAKE DECISION WITHOUT CONTRARIAN]"
            contrarian_conf = 0
            
        judge_prompt = (
            f"You are the Supreme Judge. {ppo_signal} on {target}. \n"
            f"BULL ({bull_conf}%): '{bull_reply[:50]}'\nBEAR ({bear_conf}%): '{bear_reply[:50]}'\n"
            f"RISK ({risk_score}/10): '{risk_reply[:50]}'\nCONTRARIAN: '{contrarian_reply[:50]}'\n"
            f"Rules: If Risk Score >= 7, you MUST veto. Output ONLY [APPROVED] or [VETO] followed by reasoning. Do not complain if an agent is OFFLINE."
        )
        judge_reply = self._query_llm(judge_prompt)
        
        # ═══════════════════════════════════════════
        # VERDICT COMPUTATION (Weighted Council Vote)
        # ═══════════════════════════════════════════
        # Hard rules override LLM judgment for safety
        if risk_score >= 8:
            verdict = "[VETO]"
            logger.error(f"⛔ [SUPREME JUDGE]: RISK OVERRIDE — Risk score {risk_score}/10 exceeds safety threshold. [VETO]")
        elif risk_score >= 7 and bear_conf > bull_conf:
            verdict = "[VETO]"
            logger.error(f"⛔ [SUPREME JUDGE]: High risk ({risk_score}/10) + Bear dominance ({bear_conf}% > {bull_conf}%). [VETO]")
        elif "[APPROVED]" in judge_reply.upper():
            verdict = "[APPROVED]"
            logger.success(f"👨‍⚖️ [SUPREME JUDGE]: Council vote passes. Trade authorized. [APPROVED]")
        else:
            verdict = "[VETO]"
            logger.error(f"⛔ [SUPREME JUDGE]: Council rejected the proposal. [VETO]")
        
        # Save full debate transcript to logs
        try:
             log_data = {
                 "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "target": target,
                 "signal": ppo_signal,
                 "bull": bull_reply,
                 "bull_confidence": bull_conf,
                 "bear": bear_reply,
                 "bear_confidence": bear_conf,
                 "risk_analyst": risk_reply,
                 "risk_score": risk_score,
                 "contrarian": contrarian_reply,
                 "judge": verdict
             }
             os.makedirs("logs", exist_ok=True)
             with open("logs/swarm_logs.json", "a", encoding="utf-8") as f:
                 f.write(json.dumps(log_data) + "\n")
        except Exception:
             pass
             
        return verdict

    def _extract_confidence(self, text: str, keyword: str = "CONFIDENCE") -> int:
        """Extracts confidence percentage from LLM output."""
        try:
            match = re.search(rf'{keyword}:\s*(\d+)', text, re.IGNORECASE)
            if match:
                return min(int(match.group(1)), 100)
        except Exception:
            pass
        return 50  # Default neutral confidence
    
    def _extract_risk_score(self, text: str) -> int:
        """Extracts risk score (1-10) from Risk Analyst output."""
        try:
            match = re.search(r'RISK_SCORE:\s*(\d+)', text, re.IGNORECASE)
            if match:
                return min(int(match.group(1)), 10)
        except Exception:
            pass
        return 5  # Default moderate risk

    def _query_llm(self, prompt: str) -> str:
        """Helper to fetch completions from Gemini strictly for production with timeout."""
        try:
             # Apply strict timeout/latency cap
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
             t.join(timeout=8.0) # 8 second fallback timeout for latency safety
             
             if t.is_alive():
                 logger.error("LLM API Call Timeout Exceeded (8s). Bailing out.")
                 return "TIMEOUT_FALLBACK_RISK: SAFE VETO PREFERRED. CONFIDENCE: 0%. RISK_SCORE: 10/10."
                 
             if response:
                 return response.text.strip().replace('\n', ' ')
             return "ERROR_FALLBACK"
             
        except Exception as e:
             logger.error(f"LLM API Call Failed: {e}")
             return "ERROR_FALLBACK: SAFE VETO PREFERRED. CONFIDENCE: 0%. RISK_SCORE: 10/10."
