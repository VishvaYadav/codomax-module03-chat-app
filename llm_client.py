"""
llm_client.py — a small, robust wrapper around the Google Gemini API.

Covers the Module 03 concepts:
  * Authentication & environment variables  -> get_api_key()
  * Conversation history & message roles    -> to_contents()
  * Model parameters & token usage          -> GeminiChat.stream_reply(), Usage
  * Structured (JSON) responses             -> GeminiChat.summarize_chat()
  * Error handling, retries, cost tracking  -> friendly_error(), Usage.cost()
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, asdict

from google import genai
from google.genai import types
from pydantic import BaseModel

# Free-tier models (Google renames models often; the "-latest" aliases follow the newest Flash).
MODELS = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

# Paid-tier prices in USD per 1M tokens, used ONLY to show what the chat *would* cost.
# On the free tier you pay $0. Edit these to match ai.google.dev/pricing.
DEFAULT_PRICES = {"input": 0.30, "output": 2.50}

REQUEST_TIMEOUT_S = 60  # give up on a single request after this many seconds


# --------------------------------------------------------------------------- #
# 1. Authentication                                                           #
# --------------------------------------------------------------------------- #
def get_api_key() -> str | None:
    """Look for the key in: Streamlit secrets -> .env file -> environment variable.
    Never hard-code API keys in source code."""
    try:
        import streamlit as st
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass  # not running in Streamlit, or no secrets.toml
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.environ.get("GEMINI_API_KEY")


# --------------------------------------------------------------------------- #
# 2. Errors                                                                   #
# --------------------------------------------------------------------------- #
class LLMError(Exception):
    """An error with a message that is safe and helpful to show to users."""

    def __init__(self, user_message: str, retryable: bool = False, detail: str = ""):
        super().__init__(user_message)
        self.user_message = user_message
        self.retryable = retryable
        self.detail = detail  # technical details for debugging


def friendly_error(e: Exception) -> LLMError:
    if isinstance(e, LLMError) or hasattr(e, "user_message"):
        return e
    err = _classify(e)
    err.detail = f"{type(e).__name__}: {str(e)[:500]}"
    return err


def _classify(e: Exception) -> LLMError:
    code = getattr(e, "code", None)
    text = str(e)
    if code == 429 or "RESOURCE_EXHAUSTED" in text:
        return LLMError("⏳ Free-tier rate limit reached. Wait about a minute and try again.", retryable=True)
    if code in (401, 403) or "API_KEY_INVALID" in text or "API key not valid" in text:
        return LLMError("🔑 The API key is invalid or not allowed. Check GEMINI_API_KEY.")
    if code == 404:
        return LLMError("🤖 This model isn't available for your key. Pick another model in the sidebar.")
    if code == 400:
        return LLMError(f"⚠️ The request was rejected (400): {text[:200]}")
    if code is not None and code >= 500:
        return LLMError("🛠️ Gemini servers are busy. Please try again.", retryable=True)
    if "timeout" in type(e).__name__.lower() or "timed out" in text.lower() or "timeout" in text.lower():
        return LLMError(f"⌛ Gemini didn't answer within {REQUEST_TIMEOUT_S}s. Try again, or pick a "
                        "Flash-Lite model in the sidebar (faster).")
    if "connect" in type(e).__name__.lower() or "connection" in text.lower():
        return LLMError("🌐 Network problem while contacting Gemini. Please try again.", retryable=True)
    return LLMError(f"Unexpected error: {text[:200]}")


# --------------------------------------------------------------------------- #
# 3. Token usage & cost                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0       # includes "thinking" tokens, which are billed as output
    latency_s: float = 0.0
    finish_reason: str = ""

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost(self, prices: dict = DEFAULT_PRICES) -> float:
        return (self.input_tokens * prices["input"] + self.output_tokens * prices["output"]) / 1_000_000

    def record(self, usage_metadata) -> None:
        if not usage_metadata:
            return
        self.input_tokens = usage_metadata.prompt_token_count or 0
        self.output_tokens = (usage_metadata.candidates_token_count or 0) + \
                             (getattr(usage_metadata, "thoughts_token_count", 0) or 0)

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# 4. Conversation history -> Gemini format                                    #
# --------------------------------------------------------------------------- #
def to_contents(history: list[dict], max_turns: int | None = None) -> list[types.Content]:
    """Convert [{"role": "user"|"assistant", "content": str}, ...] into Gemini Contents.

    Gemini uses the roles "user" and "model" (= assistant). The *system* message is not
    part of the history; it is sent separately as `system_instruction`.
    `max_turns` keeps only the last N user+assistant pairs, which limits tokens and cost.
    """
    msgs = history[-max_turns * 2:] if max_turns else list(history)
    while msgs and msgs[0]["role"] != "user":  # history must start with a user message
        msgs = msgs[1:]
    return [
        types.Content(role="user" if m["role"] == "user" else "model",
                      parts=[types.Part(text=m["content"])])
        for m in msgs
    ]


# --------------------------------------------------------------------------- #
# 5. Structured output schema                                                 #
# --------------------------------------------------------------------------- #
class ChatSummary(BaseModel):
    title: str
    summary: str
    key_points: list[str]
    action_items: list[str]
    topics: list[str]


# --------------------------------------------------------------------------- #
# 6. The client                                                               #
# --------------------------------------------------------------------------- #
class GeminiChat:
    def __init__(self, api_key: str | None, model: str = MODELS[0]):
        if not api_key:
            raise LLMError("🔑 No API key found. Add GEMINI_API_KEY to Streamlit secrets or a .env file.")
        # 60 s time limit per request; SDK's own retries disabled because we retry ourselves
        # (hidden SDK retries can make the app look frozen for minutes).
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_S * 1000,
            retry_options=types.HttpRetryOptions(attempts=1)))
        self.model = model

    def list_chat_models(self) -> list[str]:
        """Ask the API which text-chat models this key can use (Google renames models often).
        Flash models first (best for the free tier), then Flash-Lite, then the rest."""
        skip = ("image", "tts", "audio", "live", "embedding", "robotics", "computer-use", "veo", "imagen")
        names = set()
        for m in self.client.models.list():
            name = (m.name or "").removeprefix("models/")
            actions = getattr(m, "supported_actions", None) or []
            if "generateContent" in actions and name.startswith("gemini") and not any(s in name for s in skip):
                names.add(name)

        def version(n):  # "gemini-3.8-flash" -> 3.8 ; aliases like "gemini-flash-latest" -> 0
            m = re.match(r"gemini-(\d+(?:\.\d+)?)", n)
            return float(m.group(1)) if m else 0.0

        def rank(n):  # stable Flash first, newest version first, then Flash-Lite, then the rest
            tier = 0 if ("flash" in n and "lite" not in n) else 1 if "flash-lite" in n else 2
            return (tier, "preview" in n or "exp" in n, -version(n), n)
        return sorted(names, key=rank)

    @staticmethod
    def _config(system=None, temperature=1.0, top_p=0.95, max_tokens=2048, schema=None):
        cfg = {"temperature": temperature, "top_p": top_p, "max_output_tokens": max_tokens}
        if system:
            cfg["system_instruction"] = system
        if schema is not None:
            cfg["response_mime_type"] = "application/json"
            cfg["response_schema"] = schema
        return types.GenerateContentConfig(**cfg)

    def stream_reply(self, history, system=None, temperature=1.0, top_p=0.95,
                     max_tokens=2048, max_turns=10, usage: Usage | None = None, retries=2):
        """Yield the reply chunk by chunk (streaming). Fills `usage` when finished.
        Retries automatically on rate limits / server errors, but only if nothing was
        streamed yet (so the user never sees duplicated text)."""
        usage = usage if usage is not None else Usage()
        contents = to_contents(history, max_turns)
        config = self._config(system, temperature, top_p, max_tokens)
        for attempt in range(retries + 1):
            started, t0 = False, time.time()
            try:
                for chunk in self.client.models.generate_content_stream(
                        model=self.model, contents=contents, config=config):
                    usage.record(chunk.usage_metadata)
                    if chunk.candidates and chunk.candidates[0].finish_reason:
                        usage.finish_reason = str(chunk.candidates[0].finish_reason)
                    if chunk.text:
                        started = True
                        yield chunk.text
                usage.latency_s = round(time.time() - t0, 2)
                return
            except Exception as e:
                err = friendly_error(e)
                if started or not err.retryable or attempt == retries:
                    raise err from e
                time.sleep(3 * (attempt + 1))  # simple back-off

    def reply(self, history, system=None, temperature=1.0, top_p=0.95,
              max_tokens=2048, max_turns=10, retries=2) -> tuple[str, Usage]:
        """Non-streaming reply: wait for the full answer, then return (text, usage)."""
        contents = to_contents(history, max_turns)
        config = self._config(system, temperature, top_p, max_tokens)
        usage = Usage()
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=config)
                usage.record(resp.usage_metadata)
                usage.latency_s = round(time.time() - t0, 2)
                if resp.candidates and resp.candidates[0].finish_reason:
                    usage.finish_reason = str(resp.candidates[0].finish_reason)
                return resp.text or "", usage
            except Exception as e:
                err = friendly_error(e)
                if not err.retryable or attempt == retries:
                    raise err from e
                time.sleep(3 * (attempt + 1))

    def summarize_chat(self, history, retries=2) -> tuple[ChatSummary, Usage]:
        """Structured response: returns a validated ChatSummary object (JSON under the hood)."""
        transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
        prompt = ("Summarize this conversation. Keep key_points and action_items short. "
                  "Use an empty list if there are no action items.\n\n" + transcript)
        usage = Usage()
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt,
                    config=self._config(temperature=0.2, max_tokens=4096, schema=ChatSummary))
                usage.record(resp.usage_metadata)
                usage.latency_s = round(time.time() - t0, 2)
                parsed = resp.parsed if isinstance(resp.parsed, ChatSummary) \
                    else ChatSummary.model_validate_json(resp.text)
                return parsed, usage
            except Exception as e:
                err = friendly_error(e)
                if not err.retryable or attempt == retries:
                    raise err from e
                time.sleep(3 * (attempt + 1))
