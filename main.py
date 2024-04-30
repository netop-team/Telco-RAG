from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json

from Telco-RAG import TelcoRAG 

app = FastAPI()

# Setup CORS policy for the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryData(BaseModel):
    query: str
    api_key: str

@app.post("/process_query/")
async def process_query(data: QueryData):
    """Processes incoming queries using the TelcoRAG."""
    try:
        # Configure environment for API key and duplicate library handling flag
        os.environ["OPENAI_API_KEY"] = data.api_key
        os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
        
        # Generate response using the TelcoRAG model
        response, retrieval, query = TelcoRAG(data.query, data.api_key)

        return json.dumps({"result": response, "retrieval": retrieval, "query": query})
    except Exception as e:
        # Catch any exception and return an HTTP 500 response
        raise HTTPException(status_code=500, detail=str(e))
