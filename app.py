"""
AI Chat Application: Codomax Internship, Module 03
Streamlit + Google Gemini API

Run locally:   streamlit run app.py
"""

import json
from datetime import datetime

import streamlit as st

from llm_client import (DEFAULT_PRICES, MODELS, GeminiChat, LLMError, Usage,
                        get_api_key)

st.set_page_config(page_title="Gemini Chat · Codomax", page_icon="💬", layout="centered")

# ---- Limits that protect the free API quota on a public website -------------
MAX_USER_MESSAGES = 30          # per browser session
SESSION_TOKEN_BUDGET = 100_000  # per browser session

PERSONAS = {
    "🤖 Helpful assistant": "You are a helpful, friendly assistant. Give clear, accurate answers. "
                           "Use short paragraphs and bullet points when helpful.",
    "👨‍🏫 Coding tutor": "You are a patient Python tutor for engineering students. Explain step by step, "
                       "show small code examples, and end with one practice question.",
    "💼 Interview coach": "You are a technical interview coach. Ask one question at a time, wait for the "
                         "answer, then give honest feedback and a model answer.",
    "📚 Study buddy": "You are a study buddy who explains AI/ML concepts with simple analogies, "
                     "then checks understanding with a quick question.",
    "✏️ Custom": "",
}

STARTERS = [
    "Explain tokens and context windows in simple words",
    "Write a Python function to reverse a string, with tests",
    "Give me 3 tips for my first technical interview",
]

# ---- Session state ----------------------------------------------------------
ss = st.session_state
ss.setdefault("messages", [])  # [{"role", "content", "usage"}]
ss.setdefault("totals", {"input": 0, "output": 0, "requests": 0})
ss.setdefault("summary", None)


@st.cache_resource(show_spinner=False)
def get_client(api_key: str, model: str) -> GeminiChat:
    return GeminiChat(api_key, model)


@st.cache_data(ttl=3600, show_spinner=False)
def available_models(api_key: str) -> list[str]:
    """Models this key can actually use, fetched from the API (falls back to a fixed list)."""
    try:
        return GeminiChat(api_key).list_chat_models() or MODELS
    except Exception:
        return MODELS


def add_usage(u: Usage):
    ss.totals["input"] += u.input_tokens
    ss.totals["output"] += u.output_tokens
    ss.totals["requests"] += 1


# ---- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = get_api_key()
    if not api_key:
        api_key = st.text_input("Gemini API key", type="password",
                                help="Free key from aistudio.google.com. Kept only in this browser session.")

    model_options = available_models(api_key) if api_key else MODELS
    model = st.selectbox("Model", model_options,
                         help="Fetched live from Google: models your key can use. Flash models are best on the free tier.")

    st.subheader("🧭 Custom instructions")
    persona = st.selectbox("Persona", list(PERSONAS))
    system_prompt = st.text_area("System instruction", value=PERSONAS[persona], height=130,
                                 key=f"sys_{persona}",
                                 placeholder="e.g. Answer like a pirate. Keep replies under 50 words.")

    with st.expander("🎛️ Model parameters"):
        temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1,
                                help="Higher = more creative and varied, lower = more focused.")
        top_p = st.slider("Top-p", 0.1, 1.0, 0.95, 0.05,
                          help="Only sample from the most likely tokens whose probabilities add up to p.")
        max_tokens = st.slider("Max output tokens", 256, 8192, 2048, 256,
                               help="Upper limit on reply length. Lower = cheaper, but replies may be cut off.")
        max_turns = st.slider("Memory (last N exchanges sent)", 1, 20, 8,
                              help="Older messages are not sent to the model. This saves tokens and cost.")

    st.subheader("📊 Usage & cost")
    t = ss.totals
    c1, c2 = st.columns(2)
    c1.metric("Input tokens", f"{t['input']:,}")
    c2.metric("Output tokens", f"{t['output']:,}")
    c1.metric("Requests", t["requests"])
    used = t["input"] + t["output"]
    paid_cost = Usage(t["input"], t["output"]).cost(DEFAULT_PRICES)
    c2.metric("Your cost", "₹0", help="Free tier. No billing account is linked.")
    st.caption(f"On a paid plan this chat would cost ≈ ${paid_cost:.4f} "
               f"(at ${DEFAULT_PRICES['input']}/1M input, ${DEFAULT_PRICES['output']}/1M output tokens).")
    st.progress(min(used / SESSION_TOKEN_BUDGET, 1.0),
                text=f"Session budget: {used:,} / {SESSION_TOKEN_BUDGET:,} tokens")

    st.divider()
    b1, b2 = st.columns(2)
    if b1.button("🗑️ Clear chat", width="stretch"):
        ss.messages, ss.summary = [], None
        ss.totals = {"input": 0, "output": 0, "requests": 0}
        st.rerun()
    summarize_clicked = b2.button("🧾 Summarize", width="stretch",
                                  disabled=len(ss.messages) < 2,
                                  help="Structured JSON summary of the conversation")
    st.download_button(
        "⬇️ Download chat (JSON)", width="stretch", disabled=not ss.messages,
        file_name=f"chat_{datetime.now():%Y%m%d_%H%M}.json", mime="application/json",
        data=json.dumps({"model": model, "system_instruction": system_prompt,
                         "messages": ss.messages, "totals": ss.totals}, indent=2))

# ---- Main area --------------------------------------------------------------
st.title("💬 Gemini Chat")
st.caption("AI chat app with history, custom instructions and token tracking · "
           "Codomax GenAI Internship · Module 03")

client = None
if api_key:
    try:
        client = get_client(api_key, model)
    except LLMError as e:
        st.error(e.user_message)
else:
    st.info("👈 Add your free Gemini API key in the sidebar to start chatting.")

# Structured summary (Objective 3)
if summarize_clicked and client:
    with st.spinner("Creating structured summary..."):
        try:
            summary, u = client.summarize_chat(ss.messages)
            add_usage(u)
            ss.summary = summary.model_dump()
        except LLMError as e:
            st.error(e.user_message)

if ss.summary:
    s = ss.summary
    with st.container(border=True):
        st.subheader(f"🧾 {s['title']}")
        st.write(s["summary"])
        if s["key_points"]:
            st.markdown("**Key points**\n" + "\n".join(f"- {p}" for p in s["key_points"]))
        if s["action_items"]:
            st.markdown("**Action items**\n" + "\n".join(f"- [ ] {a}" for a in s["action_items"]))
        if s["topics"]:
            st.caption("Topics: " + ", ".join(s["topics"]))
        with st.expander("Raw JSON"):
            st.json(s)
        if st.button("Hide summary"):
            ss.summary = None
            st.rerun()

# Empty-state starter prompts
if not ss.messages:
    st.markdown("##### Try one of these:")
    cols = st.columns(len(STARTERS))
    for col, text in zip(cols, STARTERS):
        if col.button(text, width="stretch"):
            ss.pending = text
            st.rerun()

# Chat history
for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("usage"):
            u = m["usage"]
            st.caption(f"🔢 {u['input_tokens']} in · {u['output_tokens']} out · ⏱️ {u['latency_s']}s")

# New input
prompt = st.chat_input("Type your message...", disabled=client is None) or ss.pop("pending", None)

if prompt and client:
    n_user = sum(m["role"] == "user" for m in ss.messages)
    if n_user >= MAX_USER_MESSAGES:
        st.warning(f"Message limit reached ({MAX_USER_MESSAGES}) for this session. Clear the chat to start over.")
    elif ss.totals["input"] + ss.totals["output"] >= SESSION_TOKEN_BUDGET:
        st.warning("Session token budget used up. Clear the chat to start over.")
    else:
        ss.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            usage = Usage()
            try:
                reply = st.write_stream(client.stream_reply(
                    ss.messages, system=system_prompt or None, temperature=temperature,
                    top_p=top_p, max_tokens=max_tokens, max_turns=max_turns, usage=usage))
                if not isinstance(reply, str):
                    reply = "".join(map(str, reply))
                if not reply.strip():
                    reply = "_(The model returned an empty reply. Try increasing **Max output tokens**.)_"
                if "MAX_TOKENS" in usage.finish_reason:
                    reply += "\n\n_✂️ Reply cut off by the max-output-tokens limit._"
                ss.messages.append({"role": "assistant", "content": reply, "usage": usage.to_dict()})
                add_usage(usage)
                st.rerun()
            except LLMError as e:
                ss.messages.pop()  # remove the unanswered user message so they can retry
                st.error(e.user_message)
