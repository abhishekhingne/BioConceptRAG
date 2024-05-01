from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from load_document import LoadDocument
from simple_rag import SimpleRAG
from advanced_rag import AdvancedRAG

# Create an instance of the FastAPI class
app = FastAPI()

# Define a Pydantic model for request body (if needed)
class RAG(BaseModel):
    query: str
    llm_url: str

# Define simple rag POST endpoint
@app.post("/simple_rag/")
def simple_rag(payload: RAG):
    load_document = LoadDocument(llm_url=payload.llm_url, file_path="./")
    vector_store = load_document.load_documents(chunk_size=2048, chunk_overlap=200, use_existing=True)
    retriever = load_document.retrive_documents(vector_store=vector_store, k=3)
    rag = SimpleRAG(llm_url=payload.llm_url)
    result = rag.rag(query=payload.query, retriever=retriever)
    return result

# Define advance rag POST endpoint
@app.post("/advanced_rag/")
def advance_rag(payload: RAG):
    load_document = LoadDocument(llm_url=payload.llm_url, file_path="./")
    vector_store = load_document.load_documents(chunk_size=2048, chunk_overlap=200, use_existing=True)
    rag = AdvancedRAG(llm_url=payload.llm_url, vector_store=vector_store)
    result = rag.execute_graph(question=payload.query)
    keys = ["input", "answer", "context"]
    for k, n_key in zip(result.keys(), keys):
        result[n_key] = result.pop(k)
    return result