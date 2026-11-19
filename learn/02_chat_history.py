"""
02 · Conversation history and message roles (system / user / assistant)
Run:  python learn/02_chat_history.py   (type 'quit' to exit)

LLMs are stateless: they remember nothing between calls.
"Memory" = we send the previous messages again with every request.
  system    -> how the assistant should behave (sent as system_instruction)
  user      -> what the human says
  assistant -> what the model said before ("model" role in Gemini)
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM = "You are a friendly tutor. Keep answers under 60 words."
history: list[types.Content] = []

while True:
    user_text = input("\nYou: ").strip()
    if user_text.lower() in {"quit", "exit"}:
        break
    history.append(types.Content(role="user", parts=[types.Part(text=user_text)]))

    resp = client.models.generate_content(
        model="gemini-flash-latest",
        contents=history,                                   # the WHOLE conversation
        config=types.GenerateContentConfig(system_instruction=SYSTEM),
    )
    print("AI:", resp.text)
    history.append(types.Content(role="model", parts=[types.Part(text=resp.text)]))
    print(f"   (messages in history: {len(history)}, input tokens this call: "
          f"{resp.usage_metadata.prompt_token_count})")

# Input tokens grow with every turn because the history is resent each time.
# That is why the app limits memory to the last N exchanges.
