from typing import Dict, Any
import json
import requests
from core.llm_client import llm_client
from core.config import settings
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyst:
    @staticmethod
    def fetch_news(symbol: str) -> list:
        if not settings.NEWS_API_KEY:
            return ["No NewsAPI key configured, using simulated neutral headline."]
        try:
            # Map symbol to general keyword (e.g., EURUSD to EUR or Forex)
            keyword = "forex" if "USD" in symbol else symbol
            url = f"https://newsapi.org/v2/everything?q={keyword}&sortBy=publishedAt&pageSize=5&apiKey={settings.NEWS_API_KEY}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                articles = response.json().get('articles', [])
                return [a['title'] for a in articles if a.get('title')]
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
        return ["Failed to fetch live news."]

    @staticmethod
    def analyze(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uses NewsAPI and Gemini 1.5 Flash to analyze news sentiment.
        """
        symbol = state['symbol']
        logger.info(f"Sentiment Analyst running for {symbol}")
        
        headlines = SentimentAnalyst.fetch_news(symbol)
        
        prompt = f"""
        You are a financial sentiment analyst.
        Analyze the following recent news headlines for {symbol} and generate a structured JSON report.
        
        Headlines: {headlines}
        
        Output MUST be valid JSON with the following structure:
        {{
            "agent": "SENTIMENT_ANALYST",
            "symbol": "{symbol}",
            "finbert_scores": {{"positive": 0.68, "neutral": 0.22, "negative": 0.10}},
            "net_sentiment": 0.58,
            "sentiment_label": "BULLISH",
            "key_headlines": ["headline 1", "headline 2"],
            "gemini_interpretation": "Detailed interpretation",
            "sentiment_strength": "MODERATE",
            "already_priced_in": false
        }}
        """
        
        try:
            model = llm_client.get_flash_model()
            response_json = llm_client.generate_json_response(model, prompt)
            state["sentiment_report"] = json.loads(response_json)
        except Exception as e:
            logger.error(f"Sentiment Analyst Error: {e}")
            state["sentiment_report"] = {"error": str(e), "net_sentiment": 0}
            
        return state
