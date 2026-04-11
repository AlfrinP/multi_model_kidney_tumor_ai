import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file if present

def get_env_var(key: str, default=None, required=False):
    value = os.getenv(key, default)
    if required and value is None:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return value
