import runpod
import pickle
import pandas as pd
import numpy as np
import traceback

# Load the brain into memory when the container starts
print("Loading Kronos AI model...")
with open('kronos_model.pkl', 'rb') as f:
    model = pickle.load(f)
print("Model loaded successfully!")

def handler(event):
    """
    This handles incoming API requests from RunPod.
    The input data should be passed in the 'input' dictionary.
    """
    try:
        input_data = event["input"]
        
        # Convert dictionary to DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure we have the exact 8 features
        features = ['open', 'high', 'low', 'close', 'tick_volume', 'rsi', 'bb_width', 'dist_sma20']
        X = df[features]
        
        # Get predictions
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        
        return {
            "status": "success",
            "prediction": prediction,
            "probability": probability
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

# Start the RunPod serverless handler
if __name__ == "__main__":
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
        
    runpod.serverless.start({"handler": handler})
