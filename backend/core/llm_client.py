import google.generativeai as genai
from core.config import settings
import logging

logger = logging.getLogger(__name__)

# Configure the API key
genai.configure(api_key=settings.GEMINI_API_KEY)

class LLMClient:
    """Wrapper for Google Gemini API."""
    
    @staticmethod
    def get_pro_model():
        """Returns the gemini-2.5-pro model (Best for reasoning)."""
        return genai.GenerativeModel('gemini-2.5-pro')
        
    @staticmethod
    def get_flash_model():
        """Returns the gemini-2.5-flash model (Fastest for quick tasks)."""
        return genai.GenerativeModel('gemini-2.5-flash')
        
    @staticmethod
    def generate_json_response(model, prompt: str) -> str:
        """
        Helper to force the model to return JSON.
        """
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json"
            )
        )
        return response.text

llm_client = LLMClient()
