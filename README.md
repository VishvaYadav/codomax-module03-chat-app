# 💬 Gemini Chat: AI Chat Application

**Codomax Digital Solutions · Generative AI Internship · Module 03 (Day 11 – Day 15)**
Intern: **Vishva Yadav** · B.E. Computer Engineering

🌐 **Live app:** [your-app-link.streamlit.app](https://your-app-link.streamlit.app)

An AI chat web app built with **Python, Streamlit and the Google Gemini API**. It has streaming replies, conversation memory,
custom instructions (personas), adjustable model parameters, token and cost tracking, structured JSON summaries, and
friendly error handling.

## ✨ Features

| Feature | How it works |
|---|---|
| 💬 Chat with streaming | Replies appear word by word (`generate_content_stream`) |
| 🧠 Conversation history | Previous messages are re-sent with each request, using the `user` / `model` roles |
| 🧭 Custom instructions | Persona presets plus an editable **system instruction** |
| 🎛️ Model parameters | Temperature, top-p, max output tokens, memory window |
| 📊 Token usage & cost | Input/output tokens per message and per session, plus an estimated paid-tier cost |
| 🧾 Structured response | **Summarize** button returns validated JSON (Pydantic schema) |
| 🛡️ Error handling | Friendly messages for invalid key, rate limit, wrong model and server errors, with automatic retries and back-off |
| 💸 Cost control | Memory window, max-token limit, per-session message limit and token budget |
| ⬇️ Export | Download the whole chat as JSON |

## 🎯 Module objectives → code

| # | Objective | Where |
|---|---|---|
| 1 | LLM APIs, authentication, environment variables, connecting with Python | `llm_client.get_api_key()`, `learn/01_first_api_call.py` |
| 2 | Chat history, system/user/assistant messages, parameters, token usage | `llm_client.to_contents()`, `Usage`, `learn/02`, `learn/03` |
| 3 | Structured responses, robust error handling, API cost management | `summarize_chat()`, `friendly_error()`, `learn/04` |
| 4 | AI Chat Application: input, response, history, custom instructions, UI | `app.py` |

## 🧠 Key concepts I learned

- **API key and environment variables:** the key is read from Streamlit Secrets or `.env`, never written in the code or pushed to GitHub.
- **LLMs are stateless:** the model forgets everything between calls. Chat "memory" means re-sending the history each time, so input tokens grow every turn.
- **Roles:** `system` sets behaviour, `user` is the human, `assistant` (`model` in Gemini) holds the previous replies.
- **Parameters:** *temperature* controls randomness, *top-p* limits sampling to the most likely tokens, *max output tokens* caps length and cost.
- **Tokens = cost:** cost = input tokens × input price + output tokens × output price. Trimming history is the easiest saving.
- **Structured output:** JSON mode plus a schema gives data a program can use reliably, not just text.
- **Error handling:** retry only temporary errors (429, 5xx), not invalid keys or bad requests.

## 📁 Project structure

```
├── app.py                 # Streamlit chat UI
├── llm_client.py          # Gemini wrapper: auth, history, streaming, usage, errors, JSON
├── learn/
│   ├── 01_first_api_call.py
│   ├── 02_chat_history.py
│   ├── 03_params_and_tokens.py
│   └── 04_structured_and_errors.py
├── .streamlit/config.toml # theme
├── .env.example           # template for the API key
└── requirements.txt
```

## 🚀 Run locally

```bash
pip install -r requirements.txt
cp .env.example .env        # then paste your free key from aistudio.google.com
streamlit run app.py
```

## ☁️ Deployment (free)

Deployed on **Streamlit Community Cloud**. The API key is stored in the app's **Secrets** (`GEMINI_API_KEY = "..."`), not in the repo.
Uses the **Gemini API free tier**, so there are no costs and no card is needed.

## 🛠️ Tech stack
Python · Streamlit · Google Gemini API (`google-genai`) · Pydantic · python-dotenv · Streamlit Community Cloud
