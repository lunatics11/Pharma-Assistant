from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pharma_knowledge"

class RAGState(TypedDict):
    question: str
    context: List[Document]
    relevance_score: int
    answer: str

def get_vector_store():
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
    return Chroma(
        collection_name = COLLECTION_NAME, 
        embedding_function = embeddings, 
        persist_directory = CHROMA_DIR,
    )

# retreival step for langgraph, in all vector DBs, searches for top 4 most similar 
def retrieve(state: RAGState):
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(state["question"], k=4)
    return {"context": docs}

# langgraph workflow that accepts shared graph state as an argument
def grade_relevance(state: RAGState):
    question = state["question"]
    docs = state["context"] # top 4
    if not docs: # no docs retrieved
        return {"relevance_score": 0}

    # combine all docs sepereate by double space lines
    context_txt = "\n\n".join(doc.page_content for doc in docs)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""
You are grading whether retreived context is useful for answering a user question. 

Question: {question}
Retrieved_context: {context_txt}
Return only one integer: 
0 = not useful
1 = partially useful
2 = useful
"""
    # send prompt to API and wait for models text response
    response = llm.invoke(prompt)
    # extract the grade 
    try:
        score = int(response.content.strip()[0])
    except Exception:
        score=0

    return {"relevance_score": score}

# langgraph again, different prompt, inside this, we used langchain functions 
def answer_from_context(state: RAGState):
    question = state["question"] # original query
    docs = state["context"] # retreived documents
    context_blocks = []
    for i,doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown source")
        context_blocks.append(f"Source {i}: {source}\n{doc.page_content}")
    context_txt = "\n\n".join(context_blocks)
    llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0)

    prompt = f"""
You are Pharma Knowledge RAG Assistant.
Rules:
1. Answer only using the provided context.
2. If the context does not support the answer, say: "I don't know from the provided sources."
3. Include source names used in the answer.
4. Be concise and business-friendly.
5. Do not invent facts.

Question:  {question}
Context:  {context_txt}
Answer:
"""
    response =  llm.invoke(prompt)
    return {"answer": response.content}

def fallback_answer(state: RAGState):
    return{
        "answer": "I don't know from the provided sources. Please add relevant KT notes, PDF, CSV, or Excel files to the data folder and re-run ingestion."
    }

# conditional router logic in langgraph
def relevance_router(state: RAGState):
    if state["relevance_score"]>=1:
        return "answer_from_context" # target node responsible for generating context
    return "fallback_answer" #generic refusal


# build graph
def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_relevance", grade_relevance)
    graph.add_node("answer_from_context", answer_from_context)
    graph.add_node("fallback_answer", fallback_answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_conditional_edges(
        "grade_relevance", 
        relevance_router, 
        {
            "answer_from_context" : "answer_from_context", 
            "fallback_answer": "fallback_answer"
        },
    )
    graph.add_edge("answer_from_context", END)
    graph.add_edge("fallback_answer", END)

    return graph.compile()


# user-facing entry point to process the question
def ask(question: str):
    graph = build_graph()
    result = graph.invoke(
        {
            "question": question, 
            "context": [], 
            "relevance_score":0, 
            "answer":"",
        }
    )
    return result["answer"]



