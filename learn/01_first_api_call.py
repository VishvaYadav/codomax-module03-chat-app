"""
01 · First API call: authentication with environment variables
Run:  python learn/01_first_api_call.py
Needs GEMINI_API_KEY in a .env file or the environment (never hard-code keys!).
"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()                                   # reads .env into environment variables
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise SystemExit("Set GEMINI_API_KEY in .env (copy .env.example → .env)")

client = genai.Client(api_key=api_key)          # authenticated client
response = client.models.generate_content(
    model="gemini-flash-latest",
    contents="Explain what an API is in two sentences.",
)

print("Reply:\n", response.text)
u = response.usage_metadata
print(f"\nTokens → input: {u.prompt_token_count}, output: {u.candidates_token_count}, total: {u.total_token_count}")
