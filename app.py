import streamlit as st
from src.rag_graph import ask

st.set_page_config(
    page_title="Pharma knowldege RAG assitant",
    page_icon="💊",
    layout="wide",
)

st.title("Pharma knowldege RAG Assitant")
st.caption("Ask questions from pharma KT notes, DOI/SOH files, PDFs, CSVs, and product notes")

with st.sidebar:
    st.header("Project 1")
    st.write("RAG+ LangChain + LangGraph + Chroma")
    st.write("Guardrail: answers only from indexed sources.")
    st.divider()
    st.write("To add knowledge:")
    st.code("Add files to /data\nRun: python src/ingest.py")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask pharma question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = ask(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})