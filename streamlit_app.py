"""
Streamlit UI for MedDocQA.

Run with:
    streamlit run streamlit_app.py

Assumes the FastAPI backend is already running (see README) - this UI
just calls it over HTTP, same as any real client would.
"""
import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="MedDocQA", page_icon="🩺", layout="centered")
st.title("🩺 MedDocQA")
st.caption("RAG-powered document Q&A with evaluated prompting strategies")

with st.sidebar:
    st.header("Settings")
    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "Add PDFs to the knowledge base", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Upload & index"):
        with st.spinner("Uploading and indexing documents..."):
            try:
                files_payload = [
                    ("files", (f.name, f.getvalue(), "application/pdf"))
                    for f in uploaded_files
                ]
                up_resp = requests.post(f"{API_URL}/upload", files=files_payload, timeout=60)
                up_resp.raise_for_status()
                st.success(f"Uploaded: {up_resp.json()['uploaded']}")

                ing_resp = requests.post(f"{API_URL}/ingest", timeout=120)
                ing_resp.raise_for_status()
                st.success(ing_resp.json())
            except Exception as e:
                st.error(f"Upload/ingest failed: {e}")

    st.divider()
    strategy = st.selectbox(
        "Prompting strategy",
        ["zero_shot", "few_shot", "structured_cot"],
        help="Compare how each strategy answers the same question",
    )
    provider = st.selectbox("LLM provider", ["gemini", "groq"])

    st.divider()
    if st.button("Re-index documents (/ingest)"):
        with st.spinner("Chunking, embedding, and loading documents..."):
            try:
                resp = requests.post(f"{API_URL}/ingest", timeout=120)
                resp.raise_for_status()
                st.success(resp.json())
            except Exception as e:
                st.error(f"Ingest failed: {e}")

question = st.text_input("Ask a question about your documents:")

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question, "strategy": strategy, "provider": provider},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            st.subheader("Answer")
            st.write(data["answer"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Latency (s)", data["latency_seconds"])
            col2.metric("Est. cost ($)", f"{data['estimated_cost_usd']:.5f}")
            col3.metric("Model", data["model"])

            st.subheader("Retrieved sources")
            st.dataframe(pd.DataFrame(data["sources"]), use_container_width=True)

        except Exception as e:
            st.error(f"Query failed: {e}")

st.divider()
st.caption(
    "Run `python -m evaluation.evaluate` to compare all strategies x "
    "providers on a fixed test set and see evaluation/results.csv"
)
