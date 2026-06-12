import os
import requests
import logging

logger = logging.getLogger(__name__)

class KronosClient:
    """
    Connects to the RunPod Serverless GPU endpoint to request Deep Learning probabilities.
    """
    def __init__(self):
        self.api_key = os.getenv("RUNPOD_API_KEY", "")
        # The unique ID of your RunPod Serverless endpoint (found in the dashboard)
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "YOUR_ENDPOINT_ID")
        self.url = f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def get_prediction(self, data):
        """
        Sends market data to RunPod and receives a probability prediction.
        """
        payload = {"input": {"data": data}}
        try:
            # Prevent failure if the endpoint isn't fully configured yet
            if self.endpoint_id == "YOUR_ENDPOINT_ID" or not self.api_key:
                logger.info("RunPod endpoint not configured, returning local fallback mock.")
                return {"direction": "BUY", "probability": 82.5}
                
            response = requests.post(self.url, headers=self.headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                # RunPod Serverless returns the handler result inside the 'output' key
                result = response.json().get('output', {})
                return result
            else:
                logger.error(f"RunPod Error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Kronos Client connection failed: {e}")
            return None
