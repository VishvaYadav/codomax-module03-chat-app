"""
04 · Structured responses + robust error handling
Run:  python learn/04_structured_and_errors.py
"""
import os, time
from dotenv import load_dotenv
from pydantic import BaseModel
from google import genai
from google.genai import types, errors

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-flash-latest"

# ---- Structured output: the model must return JSON that matches a schema
class Movie(BaseModel):
    title: str
    year: int
    genres: list[str]
    one_line_plot: str

resp = client.models.generate_content(
    model=MODEL,
    contents="Recommend 3 classic sci-fi movies.",
    config=types.GenerateContentConfig(response_mime_type="application/json",
                                       response_schema=list[Movie]),
)
for m in resp.parsed:                       # already validated Python objects
    print(f"{m.title} ({m.year}) — {', '.join(m.genres)}: {m.one_line_plot}")

# ---- Error handling with retries and exponential back-off
def safe_generate(prompt, model=MODEL, retries=3):
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=model, contents=prompt).text
        except errors.APIError as e:
            if e.code == 429 or (e.code or 0) >= 500:      # rate limit / server: retry
                wait = 2 ** attempt * 5
                print(f"  {e.code} error, retrying in {wait}s...")
                time.sleep(wait)
            else:                                         # bad key / bad request: don't retry
                return f"[Error {e.code}] {e.message}"
    return "[Gave up after retries]"

print("\nWrong model name →", safe_generate("hi", model="this-model-does-not-exist"))
print("Normal call      →", safe_generate("Say OK."))
