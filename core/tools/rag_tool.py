"""
Document Retrieval Tool for RAG functionality
Made by Rodrigo de Sarasqueta
"""

import asyncio
from typing import Dict, Any
from pydantic import Field

from .base import BaseMinimalTool
from ..vector_store import VectorStore


class DocumentRetrievalTool(BaseMinimalTool):
    """Tool for searching and retrieving documents from vector store"""
    
    name = "document_search"
    description = """
    Search for information in uploaded documents using semantic similarity.
    Use this when the user asks about content from uploaded files or needs information from the knowledge base.
    Input should be a clear search query describing what information you're looking for.
    """
    
    vector_store: VectorStore = Field(exclude=True)
    
    def __init__(self, vector_store: VectorStore, config: Dict[str, Any]):
        super().__init__(config=config, vector_store=vector_store)
    
    def _run(self, query: str) -> str:
        """Synchronous document search"""
        try:
            return asyncio.run(self._arun(query))
        except Exception as e:
            return f"Error in document search: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Asynchronous document search"""
        try:
            print(f"Document search: {query}")
            
            # Configuration
            max_results = self.config.get("max_results", 5)
            similarity_threshold = self.config.get("similarity_threshold", 0.7)
            
            # Search for similar documents
            results = await self.vector_store.search(query, n_results=max_results)
            
            if not results:
                return "No relevant documents found for this query."
            
            # Filter by similarity threshold
            filtered_results = [r for r in results if r["similarity_score"] >= similarity_threshold]
            
            if not filtered_results:
                return f"No documents found with similarity above {similarity_threshold}. Try rephrasing your query."
            
            # Format results simply
            response = "Found information:\n\n"
            
            for i, result in enumerate(filtered_results, 1):
                filename = result["metadata"].get("filename", "Unknown")
                content = result["content"][:300] + "..."
                
                response += f"{i}. {filename}: {content}\n\n"
            
            return response
            
        except Exception as e:
            error_msg = f"Error searching documents: {str(e)}"
            print(error_msg)
            return error_msg
