from dotenv import load_dotenv
load_dotenv()
"""Multi-Agent Swarm Intelligence for pre-trade debate and validation."""

import os
from loguru import logger
import openai

class SwarmDebateEngine:
    """Manages the Bear, Bull, and Judge generative personas."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
             logger.critical("Swarm Engine failure: OPENAI_API_KEY not found in environment.")
             raise ValueError("OPENAI_API_KEY is required for the LLM Debate Engine.")
        openai.api_key = self.api_key
        
    def conduct_debate(self, target: str, ppo_signal: str) -> str:
        """
        Conducts a debate between Bear and Bull on the viability of the trade,
        and outputs [APPROVED] or [VETO] based on the Judge's verdict.
        """
        logger.info(f"⚖️ SWARM ACTIVATED: Meeting Room opened to debate PPO Signal: {ppo_signal} on {target}")
        
        # 1. Bull Argument
        bull_prompt = f"You are a hyper-bullish hedge fund manager. Give exactly 1 short sentence why I must go {ppo_signal} on {target} right now."
        bull_reply = self._query_llm(bull_prompt)
        logger.info(f"🗣️ [AGENT BULL]: '{bull_reply}'")
        
        # 2. Bear Argument
        bear_prompt = f"You are a hyper-bearish short seller. Give exactly 1 short sentence why going {ppo_signal} on {target} is a guaranteed disaster."
        bear_reply = self._query_llm(bear_prompt)
        logger.info(f"🗣️ [AGENT BEAR]: '{bear_reply}'")
        
        # 3. Judge Argument
        judge_prompt = f"You are the quantitative judge. The math model suggests {ppo_signal} on {target}. Bull argues: '{bull_reply}'. Bear argues: '{bear_reply}'. You must output ONLY the word [APPROVED] or [VETO] depending on which argument is fundamentally safer to protect capital."
        judge_reply = self._query_llm(judge_prompt)
        
        if "[APPROVED]" in judge_reply.upper():
             logger.success(f"👨‍⚖️ [AGENT JUDGE]: After reviewing PPO mathematics and Swarm qualitative data, I lock this trade. [APPROVED]")
             return "[APPROVED]"
        else:
             logger.error(f"👨‍⚖️ [AGENT JUDGE]: The downside qualitative risk heavily outweighs the quantitative momentum. I am blocking execution. [VETO]")
             return "[VETO]"

    def _query_llm(self, prompt: str) -> str:
        """Helper to fetch completions from OpenAI strictly for production."""
        try:
             response = openai.ChatCompletion.create(
                 model="gpt-4o-mini",
                 messages=[{"role": "user", "content": prompt}],
                 max_tokens=60,
                 temperature=0.7
             )
             return response.choices[0].message.content.strip().replace('\n', '')
        except Exception as e:
             logger.error(f"LLM API Call Failed: {e}")
             raise e
