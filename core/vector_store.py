"""
Simple ChromaDB Vector Store for PDF ingestion and RAG
Made by Rodrigo de Sarasqueta
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import uuid


class VectorStore:
    """Simple ChromaDB-based vector store for documents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.text_splitter = None
        
        # Configuration
        vector_config = config.get("vector_store", {})
        self.db_path = vector_config.get("db_path", "./data/chroma_db")
        self.collection_name = vector_config.get("collection_name", "documents")
        
    async def initialize(self):
        """Initialize ChromaDB and embedding model"""
        print("Initializing Vector Store...")
        
        # Initialize embedding model
        await self._init_embedding_model()
        
        # Initialize text splitter
        await self._init_text_splitter()
        
        # Initialize ChromaDB
        await self._init_chromadb()
        
        print("Vector Store initialized")
    
    async def _init_embedding_model(self):
        """Initialize sentence transformer for embeddings"""
        vector_config = self.config.get("vector_store", {})
        model_name = vector_config.get("embedding_model", "all-MiniLM-L6-v2")
        
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        print(f"Embedding model loaded (dimension: {self.embedding_model.get_sentence_embedding_dimension()})")
    
    async def _init_text_splitter(self):
        """Initialize text splitter for document chunks"""
        vector_config = self.config.get("vector_store", {})
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=vector_config.get("chunk_size", 1000),
            chunk_overlap=vector_config.get("chunk_overlap", 200),
            separators=["\n\n", "\n", " ", ""]
        )
        
        print("Text splitter initialized")
    
    async def _init_chromadb(self):
        """Initialize ChromaDB client and collection"""
        # Create data directory if it doesn't exist
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        
        # Create persistent client
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"Using existing collection: {self.collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Minimal Agent document collection"}
            )
            print(f"Created new collection: {self.collection_name}")
        
        # Get collection info
        count = self.collection.count()
        print(f"Collection contains {count} documents")
    
    async def add_document_from_file(self, file_path: str, metadata: Optional[Dict] = None) -> str:
        """Add a document from file (PDF or TXT)"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"Processing document: {file_path.name}")
        
        # Load document based on file type
        if file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
        elif file_path.suffix.lower() == ".txt":
            # Simple text file loading
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Create a simple document-like object
            documents = [type('Document', (), {'page_content': content, 'metadata': {}})()]
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Combine all pages/content
        content = "\n\n".join([doc.page_content for doc in documents])
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Create metadata
        doc_metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_type": file_path.suffix.lower(),
            "document_id": doc_id
        }
        
        if metadata:
            doc_metadata.update(metadata)
        
        # Add to vector store
        await self.add_document(doc_id, content, doc_metadata)
        
        return doc_id
    
    async def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """Add document content to vector store"""
        print(f"Adding document: {doc_id}")
        
        # Split into chunks
        chunks = self.text_splitter.split_text(content)
        print(f"Split into {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks).tolist()
        
        # Create chunk IDs and metadata
        chunk_ids = []
        chunk_metadata = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_index": i,
                "chunk_count": len(chunks),
                "parent_document_id": doc_id,
                "chunk_text_length": len(chunk)
            })
            
            chunk_ids.append(chunk_id)
            chunk_metadata.append(chunk_meta)
        
        # Add to ChromaDB
        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=chunk_metadata
        )
        
        print(f"Document added: {doc_id} ({len(chunks)} chunks)")
    
    async def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        print(f"Searching: {query}")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                result = {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "chunk_id": results["ids"][0][i] if "ids" in results else None
                }
                formatted_results.append(result)
        
        print(f"Found {len(formatted_results)} results")
        return formatted_results
    
    async def delete_document(self, doc_id: str):
        """Delete all chunks of a document"""
        # Find all chunks for this document
        try:
            results = self.collection.get(
                where={"parent_document_id": doc_id}
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                print(f"Deleted document: {doc_id} ({len(results['ids'])} chunks)")
            else:
                print(f"Document not found: {doc_id}")
        except Exception as e:
            print(f"Error deleting document: {e}")
            raise
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the collection"""
        try:
            # Get all documents
            results = self.collection.get(include=["metadatas"])
            
            # Group by document ID
            documents = {}
            for metadata in results["metadatas"]:
                doc_id = metadata.get("parent_document_id")
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "file_type": metadata.get("file_type", "Unknown"),
                        "chunk_count": 0
                    }
                if doc_id:
                    documents[doc_id]["chunk_count"] += 1
            
            return list(documents.values())
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            documents = await self.list_documents()
            
            return {
                "total_chunks": count,
                "total_documents": len(documents),
                "collection_name": self.collection_name,
                "db_path": self.db_path
            }
        except Exception as e:
            return {"error": str(e)}
