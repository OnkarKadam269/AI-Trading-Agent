from typing import Dict, Any
import json
from core.llm_client import llm_client
import logging

logger = logging.getLogger(__name__)

class FundamentalAnalyst:
    @staticmethod
    def analyze(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses Gemini 1.5 Pro to analyze macro economic events.
        """
        symbol = state['symbol']
        logger.info(f"Fundamental Analyst running for {symbol}")
        
        prompt = f"""
        You are a macro-economist fundamental analyst for forex and commodities.
        Analyze the fundamental backdrop for {symbol} based on current central bank policies and economic data.
        
        Output MUST be valid JSON with the following structure:
        {{
            "agent": "FUNDAMENTAL_ANALYST",
            "macro_bias": "BULLISH",
            "confidence": 0.74,
            "key_factors": ["factor 1", "factor 2"],
            "upcoming_risks": [
                {{"event": "FOMC", "time": "18:00 UTC", "impact": "HIGH"}}
            ],
            "risk_environment": "RISK_ON",
            "recommendation": "Detailed recommendation text"
        }}
        """
        
        try:
            model = llm_client.get_pro_model()
            response_json = llm_client.generate_json_response(model, prompt)
            state["fundamental_report"] = json.loads(response_json)
        except Exception as e:
            logger.error(f"Fundamental Analyst Error: {e}")
            state["fundamental_report"] = {"error": str(e), "confidence": 0}
            
        return state
