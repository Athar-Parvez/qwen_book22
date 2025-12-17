# RAG Chatbot Integration

This document explains how to set up and integrate the RAG (Retrieval-Augmented Generation) backend with the frontend chatbot component.

## Frontend Integration

The RAG chatbot component is located at:
`src/components/rag-chatbot/RAGChatbot.jsx`

It automatically attempts to connect to a backend RAG service and falls back to simulation mode if the service is unavailable.

## Backend Requirements

To fully enable the RAG functionality, you need to implement a backend service with the following endpoints:

### 1. Query Endpoint
```
POST /api/rag/query
```

**Request Body:**
```json
{
  "query": "Your question about the textbook content",
  "top_k": 3,
  "threshold": 0.7
}
```

**Response:**
```json
{
  "response": "The generated answer to your question",
  "sources": [
    {
      "title": "Source Title",
      "text": "Relevant text from the source",
      "url": "Link to the source if applicable"
    }
  ]
}
```

### 2. Health Check Endpoint
```
GET /api/rag/health
```

**Response:**
```json
{
  "status": "ok"
}
```

## Backend Implementation (FastAPI Example)

Here's a basic implementation using FastAPI that would connect to your vector database:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    threshold: Optional[float] = 0.7

class Source(BaseModel):
    title: str
    text: str
    url: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    sources: List[Source]

@app.post("/api/rag/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    # This is where you would:
    # 1. Embed the user's query using the same technique as your stored content
    # 2. Perform vector similarity search against your textbook embeddings
    # 3. Send the most relevant chunks to an LLM with a prompt
    # 4. Return the LLM's answer that's grounded in your textbook content
    
    # Placeholder implementation
    return QueryResponse(
        response=f"Based on the textbook content, here's an answer to your question: '{request.query}'",
        sources=[
            Source(
                title="Sample Source",
                text="This is sample text from the textbook that supports the answer.",
                url="#"
            )
        ]
    )

@app.get("/api/rag/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Environment Configuration

The frontend component will attempt to connect to `/api/rag` by default. If you're hosting your backend at a different location, you can configure the frontend by:

1. Setting the `RAG_API_BASE_URL` environment variable during build
2. Or by setting `window.RAG_API_BASE_URL` in your browser environment

## Deploying with Frontend

When deploying your Docusaurus site, ensure that:

1. Your backend API is accessible from the frontend domain (considering CORS)
2. If your backend is at a different domain, configure CORS appropriately
3. The frontend can reach the backend API endpoints at the configured URL

The chatbot component will automatically handle connection issues and switch to simulation mode if needed, providing a graceful degradation experience.