FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ libgomp1 && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install runpod numpy scikit-learn xgboost

# Copy the serverless handler
COPY kronos_api.py /app/

# Copy the trained model
# Note: You MUST run train_kronos.py first so this file exists!
COPY kronos_model.pkl /app/

# Set the command to run the RunPod Serverless worker
CMD ["python", "-u", "kronos_api.py"]
