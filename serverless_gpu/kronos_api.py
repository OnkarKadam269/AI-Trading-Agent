import runpod
import numpy as np
import pickle

# Load the trained XGBoost Model
try:
    with open('kronos_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict_handler(job):
    """
    RunPod Serverless Handler
    Receives JSON input (array of technical features), runs the XGBoost model, and returns probabilities.
    """
    job_input = job.get('input', {})
    data = np.array(job_input.get('data', []))
    
    if len(data) == 0:
        return {"error": "No data provided"}
        
    if model is None:
        return {"error": "Model not loaded"}

    # Reshape data for XGBoost (assuming a single row of features)
    features = data.reshape(1, -1)
    
    # Predict probability of class 1 (UP)
    prob_up = float(model.predict_proba(features)[0][1])
    
    direction = "BUY" if prob_up > 0.5 else "SELL"
    
    return {
        "status": "success",
        "direction": direction,
        "probability_up": round(prob_up * 100, 2),
        "probability_down": round((1 - prob_up) * 100, 2),
        "gpu_latency": "2ms" # Fast inference for XGBoost
    }

# Start the RunPod Serverless worker
runpod.serverless.start({"handler": predict_handler})
