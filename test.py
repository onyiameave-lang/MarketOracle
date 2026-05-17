import os
import os
from dotenv import  load_dotenv
from pathlib import Path

env_path = Path('.') / '.env'
print(f"--- Environment Check ---")
print(f"Current Working Directory: {os.getcwd()}")
print(f"Looking for .env at:      {env_path.absolute()}")
print(f"Does .env file exist?     {env_path.exists()}")

status = load_dotenv()
print(f"load_dotenv() success:    {status}")

key = os.getenv('GEMINI_API_KEY')
print(f"Key loaded:               '{key[:5]}...{key[-5:] if key else ''}'" if key else "Key loaded: None")
