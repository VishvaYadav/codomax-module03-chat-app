"""
03 · Model parameters and token usage
Run:  python learn/03_params_and_tokens.py
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-flash-latest"
prompt = "Suggest a name for an AI study app. Name only."

# 1) Count tokens BEFORE sending (helps estimate cost)
print("Prompt tokens:", client.models.count_tokens(model=MODEL, contents=prompt).total_tokens)

# 2) Temperature: randomness
for temp in (0.0, 1.5):
    names = [client.models.generate_content(
        model=MODEL, contents=prompt,
        config=types.GenerateContentConfig(temperature=temp)).text.strip() for _ in range(3)]
    print(f"temperature={temp}: {names}")

# 3) max_output_tokens: hard limit on reply length (and cost)
resp = client.models.generate_content(
    model=MODEL, contents="Write a long essay about the history of computers.",
    config=types.GenerateContentConfig(max_output_tokens=300))
print("\nfinish_reason:", resp.candidates[0].finish_reason, "(MAX_TOKENS = cut off by the limit)")

# 4) Usage and cost estimate (paid-tier example prices; free tier = $0)
u = resp.usage_metadata
out_tokens = (u.candidates_token_count or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
PRICE_IN, PRICE_OUT = 0.30, 2.50   # USD per 1M tokens, check ai.google.dev/pricing
cost = (u.prompt_token_count * PRICE_IN + out_tokens * PRICE_OUT) / 1_000_000
print(f"input={u.prompt_token_count}  output={out_tokens}  → paid-tier cost ≈ ${cost:.6f}")
