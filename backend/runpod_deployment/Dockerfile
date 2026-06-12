FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the XGBoost model and the API script
COPY kronos_model.pkl .
COPY runpod_api.py .

# Expose the default RunPod port (though RunPod serverless handles routing)
EXPOSE 8000

# Start the RunPod Serverless handler
CMD [ "python", "-u", "/app/runpod_api.py" ]
