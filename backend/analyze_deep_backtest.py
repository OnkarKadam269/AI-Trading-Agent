import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - KRONOS DEEP ANALYST - %(message)s')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'deep_backtest_results.json')
RULES_FILE = os.path.join(os.path.dirname(__file__), 'coach_rules.json')

def analyze_deep_data():
    if not NVIDIA_API_KEY and not GEMINI_API_KEY:
        logging.error("No API KEY found! Provide NVIDIA_API_KEY or GEMINI_API_KEY in .env")
        return

    if not os.path.exists(RESULTS_FILE):
        logging.error("deep_backtest_results.json not found! Run deep_simulator.py first.")
        return

    with open(RESULTS_FILE, 'r') as f:
        trades = json.load(f)

    if len(trades) == 0:
        logging.info("No trades to analyze.")
        return

    # Calculate Summaries to prevent token limit overflows
    total_trades = len(trades)
    
    # We compress the data into a dense CSV format to fit all trades into the AI's 128k Context Window
    # Format: Pair,Dir,Prob,Result,MFE,MAE,RSI,BBW,SMA_Dist
    compressed_trades = []
    
    # Filter for interesting trades (Losses, or trades that suffered high drawdowns/reversals)
    interesting_trades = [t for t in trades if t['result'] == 'LOSS' or t['mfe_pct'] >= 0.10 or t['mae_pct'] >= 0.10]
    
    # Send up to 3,500 trades to perfectly fit into the 131,072 token limit
    for t in interesting_trades[:3500]:
        csv_line = f"{t['pair']},{t['direction']},{t['probability']:.2f},{t['result']},{t['mfe_pct']:.2f},{t['mae_pct']:.2f},{t['rsi']:.1f},{t['bb_width']:.4f},{t['dist_sma20']:.2f}"
        compressed_trades.append(csv_line)
        
    csv_payload = "\n".join(compressed_trades)

    logging.info(f"Sending {len(compressed_trades)} FULL trades (Highly Compressed) to NVIDIA AI for massive pattern recognition...")

    prompt = f"""
    You are an elite quantitative trading coach. 
    I just ran a massive tick-by-tick simulation of our XGBoost trading algorithm.
    
    Here is a massive dataset of {len(compressed_trades)} raw trades.
    The format is: Pair, Direction, AI_Probability, Result(WIN/LOSS), Max_Favorable_Excursion_%, Max_Adverse_Excursion_%, RSI, Bollinger_Band_Width, Distance_from_SMA20
    
    [DATA START]
    {csv_payload}
    [DATA END]
    
    Analyze this massive dataset. Look for mathematical correlations of why trades LOST or hit high MFE and reversed.
    Generate specific "Pre-Trade Veto Rules" to prevent entering these bad setups based on RSI, BBW, and SMA distance.
    
    CRITICAL INSTRUCTION: The `condition_python` will be evaluated BEFORE a trade is placed. Therefore, you are STRICTLY FORBIDDEN from using `mfe_pct`, `mae_pct`, `result`, or `direction` inside `condition_python`. You can ONLY use the pre-trade features: 'rsi', 'bb_width', 'dist_sma20', and 'probability'.
    
    Output EXACTLY this JSON array format and nothing else:
    [
      {{
        "rule_name": "Dynamic High Volatility Warning",
        "condition_python": "features['rsi'] > 65 and features['bb_width'] > 0.0015",
        "reason": "Historically, entering when RSI is high and bands are wide leads to an immediate reversal. Do not enter."
      }}
    ]
    """

    try:
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
        
        existing_rules = []
        if os.path.exists(RULES_FILE):
            with open(RULES_FILE, 'r') as f:
                try:
                    existing_rules = json.load(f)
                except:
                    pass
                    
        existing_rules.extend(rules)
        
        with open(RULES_FILE, 'w') as f:
            json.dump(existing_rules, f, indent=4)
            
        logging.info(f"Deep Analysis Complete! {len(rules)} new Dynamic MFE Rules generated and added to the Coach's Brain.")
        
    except Exception as e:
        logging.error(f"Deep Analysis Failed: {e}")

if __name__ == "__main__":
    analyze_deep_data()
