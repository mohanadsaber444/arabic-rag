import os

import httpx
import streamlit as st

API = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Arabic RAG", layout="wide")
st.markdown(
    """<style>
.stChatMessage p, .stChatMessage li { unicode-bidi: plaintext; text-align: start; }
</style>""",
    unsafe_allow_html=True,
)
st.session_state.setdefault("messages", [])


def call(method: str, path: str, **kwargs):
    try:
        response = httpx.request(method, f"{API}{path}", timeout=300, **kwargs)
    except httpx.HTTPError:
        st.error("Cannot reach the API. Is it running?")
        return None
    if response.status_code >= 400:
        detail = response.json() if response.content else {}
        st.error(detail.get("message", f"Request failed ({response.status_code})"))
        return None
    return response.json() if response.content else {}


with st.sidebar:
    st.header("Documents")
    collection = st.text_input("Collection", value="default")
    upload = st.file_uploader("Upload", type=["pdf", "docx", "txt", "md"])
    if upload and st.button("Ingest"):
        with st.spinner("Processing..."):
            result = call(
                "POST",
                f"/collections/{collection}/documents",
                files={"file": (upload.name, upload.getvalue())},
            )
        if result:
            st.success(f"{result['filename']} is ready (v{result['version']})")
    docs = call("GET", f"/collections/{collection}/documents") or []
    for doc in docs:
        left, right = st.columns([4, 1])
        left.write(f"{doc['filename']} v{doc['version']} ({doc['status']})")
        if right.button("Delete", key=doc["id"]):
            call("DELETE", f"/collections/{collection}/documents/{doc['id']}")
            st.rerun()

st.title("Ask your documents")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("data"):
            data = message["data"]
            if not data["grounded"]:
                st.warning("This answer could not be fully verified against the sources.")
            with st.expander(f"Sources ({len(data['sources'])})"):
                for s in data["sources"]:
                    page = f", page {s['page']}" if s["page"] else ""
                    st.markdown(f"**[{s['index']}] {s['filename']}{page}** (score {s['score']})")
                    st.text(s["text"])

if question := st.chat_input("Type your question"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            data = call("POST", f"/collections/{collection}/query", json={"question": question})
        if data:
            st.markdown(data["answer"])
            st.session_state.messages.append(
                {"role": "assistant", "content": data["answer"], "data": data}
            )
            st.rerun()
