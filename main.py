from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import threading
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from fastapi.staticfiles import StaticFiles
import os

# Create FastAPI app
app = FastAPI()

# Mount static folder for serving HTML, CSS, JS files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the pre-trained RAG pipeline
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

vector_store = FAISS.load_local(
    "C:/Users/suj33/Desktop/PharmaQA.ai/faiss_index", 
    HuggingFaceEmbeddings(model_name=embedding_model),
    allow_dangerous_deserialization=True 
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create a HuggingFace pipeline
hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Instantiate HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Retrieval-augmented generation pipeline
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Request model to handle incoming queries
class QueryRequest(BaseModel):
    query: str

# Response model to structure the response
class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]

# Query endpoint
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="No query provided!")
    
    # RAG pipeline to answer the query
    response = rag_pipeline(query)
    answer = response['result']
    sources = [doc.page_content for doc in response["source_documents"]]

    return QueryResponse(query=query, answer=answer, sources=sources)

# FastAPI server in a separate thread (useful for testing locally)
def run_fastapi():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

# FastAPI server in a separate thread
fastapi_thread = threading.Thread(target=run_fastapi)
fastapi_thread.start()
