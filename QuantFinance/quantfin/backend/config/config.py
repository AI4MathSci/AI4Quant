from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

# Load .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
logger.info(f"Loading .env file from: {env_path}")
load_dotenv(env_path)

class Settings(BaseSettings):
    """
    Configuration settings for the application. These can be set via environment variables.
    """
    app_name: str = "Quantitative Finance System"
    app_version: str = "0.1.0"  # Consider using semantic versioning
    base_url: str = "http://localhost:8000"
    
    # Reddit API credentials
    reddit_client_id: Optional[str] = os.getenv('REDDIT_CLIENT_ID')
    reddit_client_secret: Optional[str] = os.getenv('REDDIT_CLIENT_SECRET')
    reddit_user_agent: Optional[str] = os.getenv('REDDIT_USER_AGENT')

    class Config:
        """
        Configuration class for Pydantic settings.
        """
        env_prefix = "QFS_"  # Prefix for environment variables (e.g., QFS_DATABASE_URL)
        case_sensitive = True  # Make environment variables case-sensitive

# Create a single instance of the settings
config = Settings()

# Log the loaded credentials (without exposing sensitive values)
logger.info("Reddit credentials loaded:")
logger.info(f"Client ID present: {bool(config.reddit_client_id)}")
logger.info(f"Client Secret present: {bool(config.reddit_client_secret)}")
logger.info(f"User Agent present: {bool(config.reddit_user_agent)}")

if __name__ == "__main__":
    # Example usage:
    print(f"Application Name: {config.app_name}")
    print(f"Application Version: {config.app_version}")
    print(f"Base URL: {config.base_url}")
    print(f"Reddit Client ID: {config.reddit_client_id}")