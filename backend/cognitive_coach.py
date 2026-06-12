import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

import sys
sys.path.append(os.path.dirname(__file__))
from data.database import SessionLocal, Trade

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - KRONOS COACH - %(message)s')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

RULES_FILE = os.path.join(os.path.dirname(__file__), 'coach_rules.json')

def get_recent_closed_trades(limit=50):
    db = SessionLocal()
    trades = db.query(Trade).filter(Trade.exit_price != None).order_by(Trade.close_time.desc()).limit(limit).all()
    db.close()
    return trades

def analyze_and_update_rules():
    if not NVIDIA_API_KEY and not GEMINI_API_KEY:
        return []
        
    trades = get_recent_closed_trades()
    if len(trades) < 2: # We need at least a few trades to analyze
        logging.info("Not enough closed trades yet to form meaningful patterns.")
        return []
        
    # Format data for Gemini
    trade_history = []
    for t in trades:
        trade_history.append({
            "pair": t.symbol,
            "direction": t.direction,
            "pnl": t.pnl,
            "result": "WIN" if t.pnl > 0 else "LOSS",
            "indicators": t.reasoning
        })
        
    prompt = f"""
    You are an elite quantitative trading coach. 
    Analyze the following recent trades taken by our XGBoost algorithm:
    
    {json.dumps(trade_history, indent=2)}
    
    Identify specific mathematical patterns in the technical indicators (RSI, Bollinger Band Width, Distance from SMA) that caused the LOSSES.
    Do not give me advice on wins. I only want to know what conditions lead to failure.
    
    Output a JSON array of "Avoidance Rules". Each rule must be strict.
    Output EXACTLY this format and nothing else:
    [
      {{
        "rule_name": "High RSI Reversal Failure",
        "condition_python": "features['rsi'] > 70 and features['bb_width'] < 0.002",
        "reason": "When RSI is overbought but volatility is low, the breakout usually fakes out."
      }}
    ]
    Make sure the condition_python uses the dictionary `features` and uses valid python syntax.
    If there are no clear losing patterns yet, output an empty array: []
    """
    
    try:
        logging.info("Sending trades to AI for Cognitive Autopsy...")
        
        if NVIDIA_API_KEY:
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
            response = client.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1024
            )
            text = response.choices[0].message.content.strip()
        else:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-pro-exp-02-05')
            response = model.generate_content(prompt)
            text = response.text.strip()
        
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        rules = json.loads(text.strip())
        
        with open(RULES_FILE, 'w') as f:
            json.dump(rules, f, indent=4)
            
        logging.info(f"Cognitive Autopsy Complete. {len(rules)} Avoidance Rules generated and saved.")
        return rules
        
    except Exception as e:
        logging.error(f"Cognitive Coach Failed: {e}")
        return []

def load_rules():
    if os.path.exists(RULES_FILE):
        try:
            with open(RULES_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

if __name__ == "__main__":
    analyze_and_update_rules()
