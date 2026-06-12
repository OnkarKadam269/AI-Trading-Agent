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
    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']

    # Find the MFE of losses (how much profit did we have before the trade reversed and hit SL?)
    # High MFE on a loss means we should have used a Trailing Stop!
    high_mfe_losses = [t for t in losses if t['mfe_pct'] > 0.001]
    
    # Take a sample of the most painful reversals to show Gemini
    high_mfe_losses.sort(key=lambda x: x['mfe_pct'], reverse=True)
    sample_reversals = high_mfe_losses[:50]

    logging.info(f"Sending {len(sample_reversals)} deep MFE reversals to NVIDIA AI for Trailing Stop Optimization...")

    prompt = f"""
    You are an elite quantitative trading coach. 
    I just ran a massive tick-by-tick simulation of our XGBoost trading algorithm on 1 year of data.
    
    Out of {total_trades} trades, {len(losses)} hit the Stop Loss.
    However, {len(high_mfe_losses)} of those losses actually went deep into profit (High MFE - Maximum Favorable Excursion) before abruptly reversing and hitting the Stop Loss!
    
    Here is a sample of the 50 most painful reversals (trades that were winning but turned into losers):
    
    {json.dumps(sample_reversals, indent=2)}
    
    Analyze the 'mfe_pct' (how far into profit it went) and the technical indicators of these reversals.
    Generate specific "Dynamic Rules" to prevent this. For example, if you notice trades with high RSI always reverse after 0.15% profit, create a rule to VETO or warn about it.
    
    Output EXACTLY this JSON array format and nothing else:
    [
      {{
        "rule_name": "Dynamic Trailing Stop Warning",
        "condition_python": "features['rsi'] > 65 and features['bb_width'] > 0.0015",
        "reason": "Historically, these setups reverse aggressively after a small push. We must exit early."
      }}
    ]
    Make sure condition_python uses `features` dictionary.
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
