"""
app.py — PakGuide RAG Streamlit UI

Deliberately kept thin: this file has almost no logic in it. All the real
work happens in rag_pipeline.py. That separation matters — the UI shouldn't
know HOW retrieval or generation works, it just calls answer() and displays
the result. If you swap Streamlit for Gradio later, rag_pipeline.py doesn't
change at all.
"""

import streamlit as st
from rag_pipeline import answer

st.set_page_config(page_title="PakGuide", page_icon="🇵🇰")

st.title("🇵🇰 PakGuide")
st.caption("AI assistant for NADRA, FBR, HEC, and passport procedures — grounded in retrieved reference data.")

# st.session_state persists chat history across reruns (Streamlit reruns the
# whole script on every interaction, so without this, history would vanish
# after each message — this is Streamlit's version of keeping state alive).
if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources used for this answer"):
                for s in msg["sources"]:
                    st.markdown(f"**Q:** {s['instruction']}  \n*(similarity distance: {s['distance']:.3f})*")

# Chat input box
if user_query := st.chat_input("Ask about NADRA, FBR, HEC, passport procedures..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant info and generating answer..."):
            try:
                result = answer(user_query)
                st.markdown(result["answer"])
                with st.expander("Sources used for this answer"):
                    for s in result["sources"]:
                        st.markdown(f"**Q:** {s['instruction']}  \n*(similarity distance: {s['distance']:.3f})*")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })
            except Exception as e:
                error_msg = f"Something went wrong: {e}\n\nIs Ollama running? Try `ollama serve` in a terminal."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

with st.sidebar:
    st.header("How this works")
    st.markdown(
        """
        1. **Retrieve** — your question is embedded and matched against
        ChromaDB for the closest Q&A pairs.
        2. **Augment** — those pairs are injected into the prompt as context.
        3. **Generate** — your local model answers using that context.

        This is why the app can say *"I don't have that information"*
        instead of guessing — it's only trusting what retrieval actually found.
        """
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
