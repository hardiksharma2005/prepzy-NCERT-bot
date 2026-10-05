"""Streamlit chat UI for the NCERT Class 10 Science doubt-solver (talks to the FastAPI backend)."""
import os

import requests
import streamlit as st

st.set_page_config(page_title="NCERT Class 10 Science Doubt Solver", page_icon="🔬")


def backend_url() -> str:
    try:
        url = st.secrets.get("BACKEND_URL", "")
    except Exception:  # no secrets.toml locally
        url = ""
    return (url or os.getenv("BACKEND_URL", "http://localhost:8000")).rstrip("/")


API = backend_url()
# Free hosts sleep when idle; the first request after that can take ~1 minute.
TIMEOUT = 120


def new_session() -> str:
    resp = requests.post(f"{API}/session", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()["session_id"]


def send(message: str) -> dict:
    payload = {"session_id": st.session_state.session_id, "message": message}
    resp = requests.post(f"{API}/chat", json=payload, timeout=TIMEOUT)
    if resp.status_code == 404:  # backend restarted and forgot the session
        st.session_state.session_id = new_session()
        payload["session_id"] = st.session_state.session_id
        resp = requests.post(f"{API}/chat", json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def render_meta(meta: dict) -> None:
    chapters = ", ".join(meta.get("citations") or []) or "—"
    source = "⚡ Cache hit" if meta.get("cache_hit") else "🧠 Fresh answer"
    st.caption(f"📖 **Chapter:** {chapters} &nbsp;·&nbsp; {source} &nbsp;·&nbsp; "
               f"⏱️ {meta.get('latency_ms', 0)} ms")


def reset() -> None:
    st.session_state.messages = []
    st.session_state.session_id = new_session()


if "session_id" not in st.session_state:
    try:
        with st.spinner("Waking up the server (can take up to a minute the first time)..."):
            reset()
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend at {API}: {exc}")
        st.stop()

with st.sidebar:
    st.header("🔬 Class 10 Science")
    st.write("Ask any doubt from the NCERT Class 10 Science textbook. Follow-ups like "
             "*\"what about its laws?\"* or *\"explain it more simply\"* work too.")
    if st.button("🆕 New conversation", use_container_width=True):
        reset()
        st.rerun()
    st.divider()
    st.caption("**⚡ Cache hit** – a verified past answer to the same question, served "
               "instantly with no LLM call.\n\n**🧠 Fresh answer** – generated from the textbook "
               "by the LLM.")

st.title("NCERT Class 10 Science Doubt Solver")

if not st.session_state.messages:
    st.info("Try: *What is refraction?* → *What does refraction mean?* → *What about its laws?* "
            "→ *Explain it more simply*")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_meta(msg)

if prompt := st.chat_input("Ask a doubt, e.g. Why is the sky blue?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                data = send(prompt)
            msg = {"role": "assistant", "content": data["reply"], **data}
        except requests.RequestException as exc:
            msg = {"role": "assistant", "content": f"⚠️ Sorry, something went wrong: {exc}",
                   "citations": [], "cache_hit": False, "latency_ms": 0}
        st.markdown(msg["content"])
        render_meta(msg)
    st.session_state.messages.append(msg)
