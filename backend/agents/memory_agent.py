from typing import Dict, Any
import json
from core.llm_client import llm_client
import logging
import chromadb

logger = logging.getLogger(__name__)

# Initialize ChromaDB client (local persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = chroma_client.get_or_create_collection(name="trade_memory")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    collection = None

class MemoryAgent:
    @staticmethod
    def analyze(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses ChromaDB to retrieve similar historical setups and Gemini to summarize.
        """
        symbol = state['symbol']
        logger.info(f"Memory Agent running for {symbol}")
        
        # Simulate retrieval for now until we have actual embeddings and vectors to query
        similar_trades_summary = "Historically, fading news on USD with an OB works well (75% WR)."
        
        prompt = f"""
        You are the trading desk's Memory Agent.
        Analyze these historical patterns for {symbol} and generate a structured JSON report.
        
        Historical Context: {similar_trades_summary}
        Current Regime: {state.get('market_regime')}
        
        Output MUST be valid JSON with the following structure:
        {{
            "similar_setups_found": 5,
            "win_rate": 0.75,
            "avg_rr": 2.1,
            "memory_signal": "POSITIVE",
            "lessons": "What worked/failed in similar past scenarios"
        }}
        """
        
        try:
            model = llm_client.get_flash_model()
            response_json = llm_client.generate_json_response(model, prompt)
            state["memory_report"] = json.loads(response_json)
        except Exception as e:
            logger.error(f"Memory Agent Error: {e}")
            state["memory_report"] = {"error": str(e), "memory_signal": "NEUTRAL"}
            
        return state
