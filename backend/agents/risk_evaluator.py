from typing import Dict, Any
import json
from core.llm_client import llm_client
import logging

logger = logging.getLogger(__name__)

class RiskEvaluator:
    @staticmethod
    def evaluate(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates the Hybrid Confidence Score and adjusts risk using Gemini Pro.
        """
        symbol = state['symbol']
        logger.info(f"Risk Evaluator running for {symbol}")
        
        prompt = f"""
        You are the Chief Risk Officer for an AI institutional trading desk.
        Review the following synthesized analysis for {symbol} and calculate the Hybrid Confidence Score (0-100).
        
        Master Analysis: {state.get('master_analysis', {})}
        
        Output MUST be valid JSON with the following structure:
        {{
            "score": 83.7,
            "approved": true,
            "risk_pct": 1.0,
            "sl_pips": 50,
            "reasoning": "Detailed justification for the score and risk allocation."
        }}
        """
        
        try:
            model = llm_client.get_pro_model()
            response_json = llm_client.generate_json_response(model, prompt)
            state["risk_decision"] = json.loads(response_json)
        except Exception as e:
            logger.error(f"Risk Evaluator Error: {e}")
            state["risk_decision"] = {"error": str(e), "approved": False}
            
        return state
