Pharma Knowledge RAG Assistant

Goal: Build a chatbot that answers questions from pharma documents/data like inventory, SOH, DOI, sales, product notes, KT notes, etc.

Concepts covered
RAG
LangChain
LangGraph
MCP
Basic evaluation
Guardrails/corrigibility basics

Architecture

PDF/CSV/KT Notes
↓
Document Loader
↓
Chunking + Embeddings
↓
Vector DB
↓
LangGraph Workflow:
- classify query
- retrieve docs
- grade relevance
- answer with citations
- fallback if weak context

Example features
“Explain DOI for Eliquis”
“Summarize Arrotex story from KT notes”
“Find why SOH is high but DOI is low”
“Give answer only if source exists”
“Say ‘I don’t know’ when retrieval is weak”

Why this is useful: Internal case-study material shows RAG + Knowledge Graph can support contextual Q&A over biomedical knowledge, using vector search and database query generation with LangChain-style Cypher generation. AI Case Studies across value chain-ukey160625081138

# How to use
- pip install -r requirements.txt
- streamlit run app.py
