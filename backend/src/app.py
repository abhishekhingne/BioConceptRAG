from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from simple_rag import SimpleRAG

# Create an instance of the FastAPI class
app = FastAPI()

# Define a Pydantic model for request body (if needed)
class RAG(BaseModel):
    query: str
    llm_url: str

# Define your POST endpoint
@app.post("/simple_rag/")
def simple_rag(payload: RAG):
    rag = SimpleRAG(llm_url=payload.llm_url, 
                    file_path="/home/intellect/Documents/ConceptsofBiology_2chapters.pdf")
    vector_store = rag.load_documents(chunk_size=2048, chunk_overlap=200, use_existing=True)
    retriever = rag.retrive_documents(vector_store=vector_store, k=3)
    result = rag.rag(query=payload.query, retriever=retriever)
    return result