
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Configuration settings for the application.These can be set via environment variables.
    """
    app_name: str = "Quantitative Finance System"
    app_version: str = "0.1.0"  #  Consider using semantic versioning
    base_url: str = "http://localhost:8000"
    # Add other configuration variables as needed
    # For example:
    # database_url: Optional[str] = None
    # api_key: Optional[str] = None

    class Config:
        """
        Configuration class for Pydantic settings.
        """
        env_prefix = "QFS_"  # Prefix for environment variables (e.g., QFS_DATABASE_URL)
        case_sensitive = True # Make environment variables case-sensitive

# Create a single instance of the settings
config = Settings()

if __name__ == "__main__":
    # Example usage:
    print(f"Application Name: {config.app_name}")
    print(f"Application Version: {config.app_version}")
    print(f"Base URL: {config.base_url}")
    # print(f"Database URL: {config.database_url}") # example