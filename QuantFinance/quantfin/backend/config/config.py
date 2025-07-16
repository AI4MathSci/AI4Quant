from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
import logging
import time

logger = logging.getLogger(__name__)

# Load .env file
env_path = os.path.join(os.getcwd(), '.env')
logger.info(f"Loading .env file from: {env_path}")
load_dotenv(env_path)

class Settings(BaseSettings):
    """
    Configuration settings for the application. These can be set via environment variables.
    """
    app_name: str = "Quantitative Finance System"
    app_version: str = "1.0.0"  # Updated version for sentiment analysis
    base_url: str = "http://localhost:8000"
    
    # OpenAI API credentials for sentiment analysis
    openai_api_key: Optional[str] = os.getenv('OPENAI_API_KEY')
    openai_model: str = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # Class-level cache for OpenAI validation
    _openai_cache = {}
    _cache_timeout = 300  # 5 minutes cache timeout
    
    def _test_openai_connection(self) -> bool:
        """Test OpenAI API connection by listing models (with caching)"""
        if not self.openai_api_key or not self.openai_api_key.strip():
            logger.info("OpenAI API Key check: No key provided")
            return False
        
        # Check cache first
        cache_key = self.openai_api_key.strip()
        current_time = time.time()
        
        if cache_key in self._openai_cache:
            cached_result, cached_time = self._openai_cache[cache_key]
            if current_time - cached_time < self._cache_timeout:
                logger.info(f"OpenAI API Key check: Using cached result - {cached_result}")
                return cached_result
        
        try:
            import openai
            
            # Set the API key
            openai.api_key = cache_key
            
            # Test connection by listing models (free operation)
            models = openai.models.list()
            
            if models and hasattr(models, 'data') and len(models.data) > 0:
                logger.info(f"OpenAI API Key check: Valid key confirmed - {len(models.data)} models available")
                result = True
            else:
                logger.warning("OpenAI API Key check: API responded but no models available")
                result = False
                
            # Cache the result
            self._openai_cache[cache_key] = (result, current_time)
            return result
                
        except Exception as e:
            logger.warning(f"OpenAI API Key check: Connection failed - {str(e)}")
            result = False
            # Cache the failure too (but with shorter timeout)
            self._openai_cache[cache_key] = (result, current_time)
            return result
    
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is configured and valid by testing actual API connection"""
        return self._test_openai_connection()
    
    def clear_openai_cache(self):
        """Clear the cached OpenAI connection test results"""
        self._openai_cache.clear()
        logger.info("OpenAI API key cache cleared")
    
    # Legacy property names for backward compatibility
    @property
    def OPENAI_API_KEY(self) -> Optional[str]:
        return self.openai_api_key
    
    @property
    def OPENAI_MODEL(self) -> str:
        return self.openai_model

    class Config:
        """
        Configuration class for Pydantic settings.
        """
        env_prefix = "QFS_"  # Prefix for environment variables (e.g., QFS_DATABASE_URL)
        case_sensitive = True  # Make environment variables case-sensitive

# Create a single instance of the settings
config = Settings()
full_sentiment_analysis = config.has_openai_key # Boolean to indciate whether using full sentiment analysis (OpenAI API is configured)

# Sentiment Analysis Configuration
SENTIMENT_CLASSIFICATION_THRESHOLD = float(os.getenv('SENTIMENT_CLASSIFICATION_THRESHOLD', '0.1'))
SENTIMENT_CONFIDENCE_THRESHOLD = float(os.getenv('SENTIMENT_CONFIDENCE_THRESHOLD', '0.5'))
SENTIMENT_DECAY_FACTOR = float(os.getenv('SENTIMENT_DECAY_FACTOR', '0.9'))
SENTIMENT_COMBO_WEIGHT = float(os.getenv('SENTIMENT_COMBO_WEIGHT', '0.5'))
SENTIMENT_WEIGHT_DEFAULT = float(os.getenv('SENTIMENT_WEIGHT_DEFAULT', '0.3'))

# Validation function
def validate_sentiment_config():
    """Validate sentiment configuration values"""
    if not (0.0 <= SENTIMENT_CLASSIFICATION_THRESHOLD <= 1.0):
        raise ValueError(f"SENTIMENT_CLASSIFICATION_THRESHOLD must be between 0.0 and 1.0, got {SENTIMENT_CLASSIFICATION_THRESHOLD}")
    if not (0.0 <= SENTIMENT_CONFIDENCE_THRESHOLD <= 1.0):
        raise ValueError(f"SENTIMENT_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0, got {SENTIMENT_CONFIDENCE_THRESHOLD}")
    if not (0.0 <= SENTIMENT_DECAY_FACTOR <= 1.0):
        raise ValueError(f"SENTIMENT_DECAY_FACTOR must be between 0.0 and 1.0, got {SENTIMENT_DECAY_FACTOR}")
    if not (0.0 <= SENTIMENT_COMBO_WEIGHT <= 1.0):
        raise ValueError(f"SENTIMENT_COMBO_WEIGHT must be between 0.0 and 1.0, got {SENTIMENT_COMBO_WEIGHT}")
    if not (0.0 <= SENTIMENT_WEIGHT_DEFAULT <= 1.0):
        raise ValueError(f"SENTIMENT_WEIGHT_DEFAULT must be between 0.0 and 1.0, got {SENTIMENT_WEIGHT_DEFAULT}")

# Validate on import
validate_sentiment_config()

# Log the loaded credentials (without exposing sensitive values)
logger.info("Credentials loaded:")
logger.info(f"OpenAI API Key present: {bool(config.has_openai_key)}")

if __name__ == "__main__":
    # Example usage:
    print(f"Application Name: {config.app_name}")
    print(f"Application Version: {config.app_version}")
    print(f"Base URL: {config.base_url}")
    print(f"Reddit Client ID: {config.reddit_client_id}")
    print(f"OpenAI API Key present: {config.has_openai_key}")