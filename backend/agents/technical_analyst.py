from typing import Dict, Any
import json
from core.llm_client import llm_client
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyst:
    @staticmethod
    def analyze(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses Gemini to analyze the dynamic setup detected by the Quant Layer.
        Evaluates SMC, Mean Reversion, or Scalping structures.
        """
        symbol = state['symbol']
        regime = state['regime']
        setup = state.get('setup_data', {})
        
        logger.info(f"Technical Analyst running for {symbol} on {setup.get('type')} setup.")
        
        prompt = f"""
        You are an expert Institutional Technical Analyst.
        
        The Quantitative system has detected a trading setup for {symbol}.
        Market Regime: {regime}
        Setup Type: {setup.get('type')}
        Proposed Signal: {setup.get('signal')}
        Entry: {setup.get('entry')}
        Stop Loss: {setup.get('sl')}
        Take Profit: {setup.get('tp')}
        
        KRONOS DEEP LEARNING FORECAST:
        {setup.get('kronos_forecast', 'No forecast data available')}
        
        Evaluate the validity of this setup given the current market regime and the Kronos Deep Learning probabilistic forecast. Does it make sense?
        
        Output MUST be valid JSON with the following structure:
        {{
            "technical_bias": "BULLISH" or "BEARISH" or "NEUTRAL",
            "setup_quality": 0-100,
            "reasoning": "Brief explanation of why this setup is valid or invalid."
        }}
        """
        
        try:
            model = llm_client.get_flash_model()
            response_text = llm_client.generate_json_response(model, prompt)
            
            # Clean up potential markdown formatting from Gemini
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
                
            result = json.loads(response_text)
            return result
        except Exception as e:
            logger.error(f"Technical Analyst failed: {e}")
            return {
                "technical_bias": "NEUTRAL",
                "setup_quality": 0,
                "reasoning": f"Analysis failed: {str(e)}"
            }
