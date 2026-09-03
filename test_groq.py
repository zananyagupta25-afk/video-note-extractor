"""Quick diagnostic: tests Groq API connectivity directly and prints the real error."""
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.environ.get("GROQ_API_KEY", "").strip()
print(f"API key loaded: {'YES' if api_key else 'NO'} (length: {len(api_key)})")

try:
    from groq import Groq
    client = Groq(api_key=api_key)
    print("Client created. Testing a simple chat completion...")
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": "Say OK"}],
    )
    print("SUCCESS:", resp.choices[0].message.content)
except Exception as e:
    import traceback
    print("FAILED WITH FULL TRACEBACK:")
    traceback.print_exc()