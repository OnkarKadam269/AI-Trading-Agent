from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MTFConfluenceEngine:
    """
    Evaluates multi-timeframe alignment (HTF -> MTF -> LTF).
    """

    @staticmethod
    def evaluate(mtf_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Takes in analysis from multiple timeframes and scores the confluence.
        Expects keys like 'D1', 'H4', 'H1', 'M15' with their SMC bias.
        """
        logger.info("Evaluating MTF Confluence...")
        score = 0
        bias = "NEUTRAL"
        alignment_details = []
        
        # Simplified Logic
        d1_bias = mtf_data.get('D1', {}).get('bias', 'NEUTRAL')
        h4_bias = mtf_data.get('H4', {}).get('bias', 'NEUTRAL')
        h1_bias = mtf_data.get('H1', {}).get('bias', 'NEUTRAL')
        m15_bias = mtf_data.get('M15', {}).get('bias', 'NEUTRAL')
        
        if d1_bias == h4_bias == h1_bias == m15_bias and d1_bias != 'NEUTRAL':
            score = 100
            bias = d1_bias
            alignment_details.append(f"Full alignment on {bias}")
        elif h4_bias == h1_bias == m15_bias and h4_bias != 'NEUTRAL':
            score = 75
            bias = h4_bias
            alignment_details.append(f"H4 down to M15 alignment on {bias}")
        else:
            score = 30
            alignment_details.append("Mixed timeframes, no clear alignment")
            
        return {
            "score": score,
            "bias": bias,
            "alignment": "FULL" if score == 100 else "PARTIAL" if score > 50 else "NONE",
            "details": alignment_details
        }

