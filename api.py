"""
FastAPI endpoints for the minimal RAG agent
Made by Rodrigo de Sarasqueta
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import tempfile
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from config import Config
from core.agent import MinimalAgent

# Initialize FastAPI app
app = FastAPI(
    title="Minimal RAG Agent API",
    description="Simple RAG agent with multi-LLM support and document ingestion",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent: Optional[MinimalAgent] = None


# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    response: str
    success: bool
    error: Optional[str] = None
    session_id: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: str
    chunk_count: int


class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 5


class SearchResult(BaseModel):
    content: str
    filename: str
    similarity_score: float
    chunk_id: str


class StatusResponse(BaseModel):
    status: str
    llm_provider: str
    total_documents: int
    total_chunks: int
    available_tools: List[str]


@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global agent
    try:
        print("Initializing Minimal RAG Agent...")
        config = Config()
        agent = MinimalAgent(config.to_dict())
        await agent.initialize()
        print("Agent initialized successfully")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if agent else "unhealthy",
        "message": "Minimal RAG Agent API"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with the agent"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = await agent.chat(request.message)
        
        return ChatResponse(
            response=result["response"],
            success=result["success"],
            error=result.get("error"),
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or TXT document"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    # Validate file type
    allowed_types = [".pdf", ".txt"]
    file_suffix = Path(file.filename).suffix.lower()
    
    if file_suffix not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_suffix}. Allowed: {allowed_types}"
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Add document to vector store
        doc_id = await agent.add_document(
            tmp_file_path, 
            metadata={"original_filename": file.filename}
        )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return {
            "message": "Document uploaded successfully",
            "document_id": doc_id,
            "filename": file.filename
        }
        
    except Exception as e:
        # Clean up temporary file on error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all documents in the knowledge base"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        documents = await agent.vector_store.list_documents()
        return [
            DocumentInfo(
                document_id=doc["document_id"],
                filename=doc["filename"],
                file_type=doc["file_type"],
                chunk_count=doc["chunk_count"]
            )
            for doc in documents
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the knowledge base"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        await agent.vector_store.delete_document(doc_id)
        return {"message": f"Document {doc_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """Search documents in the knowledge base"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        results = await agent.vector_store.search(request.query, n_results=request.limit)
        
        return [
            SearchResult(
                content=result["content"],
                filename=result["metadata"].get("filename", "Unknown"),
                similarity_score=result["similarity_score"],
                chunk_id=result.get("chunk_id", "")
            )
            for result in results
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get agent status and statistics"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        stats = await agent.vector_store.get_stats()
        
        return StatusResponse(
            status="ready",
            llm_provider=agent.llm_client.get_provider(),
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            available_tools=agent.get_available_tools()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-memory")
async def clear_memory():
    """Clear agent memory"""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        await agent.clear_memory()
        return {"message": "Memory cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Serve minimal web interface"""
    with open("templates/index.html", "r") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
