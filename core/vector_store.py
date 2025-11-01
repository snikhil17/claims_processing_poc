"""
Vector store management for document embeddings
"""

import ollama
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store for document embeddings using ChromaDB"""
    
    def __init__(self, embedding_model: str = "nomic-embed-text", 
                 persist_directory: str = "data/chroma_db"):
        """Initialize vector store"""
        
        self.embedding_model = embedding_model
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="expense_documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"VectorStore initialized with model: {embedding_model}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        
        try:
            embeddings = []
            
            for text in texts:
                if not text or len(text.strip()) < 3:
                    # Create zero embedding for empty text
                    embeddings.append([0.0] * 768)  # Default embedding size
                    continue
                
                response = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text[:512]  # Limit text length for embedding
                )
                
                embedding = response.get('embedding', [])
                if not embedding:
                    # Fallback zero embedding
                    embedding = [0.0] * 768
                
                embeddings.append(embedding)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 768 for _ in texts]
    
    def add_document_chunks(self, document_id: str, chunks: List[str], 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add document chunks to vector store"""
        
        try:
            if not chunks:
                logger.warning(f"No chunks to add for document {document_id}")
                return False
            
            # Generate embeddings for chunks
            embeddings = self.generate_embeddings(chunks)
            
            # Create IDs for chunks
            chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Create metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                meta = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_text": chunk[:100],  # Store preview
                    "chunk_length": len(chunk)
                }
                if metadata:
                    meta.update(metadata)
                chunk_metadata.append(meta)
            
            # Add to collection
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=chunk_metadata
            )
            
            logger.info(f"Added {len(chunks)} chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding chunks to vector store: {e}")
            return False
    
    def search_similar_chunks(self, query: str, n_results: int = 5, 
                            document_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using semantic similarity"""
        
        try:
            # Generate embedding for query
            query_embeddings = self.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Prepare search filters
            where_filter = None
            if document_id:
                where_filter = {"document_id": document_id}
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, 50),  # Limit results
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    result = {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 1.0
                    }
                    formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} similar chunks for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    def query_document(self, query: str, document_id: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query for similar chunks within a specific document
        
        Args:
            query: Search query
            document_id: ID of the document to search within
            n_results: Number of results to return
            
        Returns:
            Dict with search results
        """
        
        try:
            # Generate embedding for query
            query_embeddings = self.generate_embeddings([query])
            if not query_embeddings:
                return {"documents": [], "metadatas": [], "distances": []}
            
            # Search with document filter
            results = self.collection.query(
                query_embeddings=query_embeddings[0],  # Use first embedding
                n_results=n_results,
                where={"document_id": document_id},  # Filter by document ID
                include=["documents", "metadatas", "distances"]
            )
            
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            logger.info(f"Retrieved {len(documents)} chunks for document {document_id}")
            
            return {
                "documents": documents,
                "metadatas": metadatas,
                "distances": distances
            }
            
        except Exception as e:
            logger.error(f"Error querying document {document_id}: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                # Delete chunks
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return True
            else:
                logger.info(f"No chunks found for document {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting document from vector store: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        
        try:
            count = self.collection.count()
            
            # Get sample of metadata to understand structure
            sample_results = self.collection.peek(limit=10)
            
            return {
                "total_chunks": count,
                "embedding_model": self.embedding_model,
                "collection_name": self.collection.name,
                "persist_directory": str(self.persist_directory),
                "sample_metadata": sample_results.get("metadatas", [])[:3] if sample_results else []
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all data from the collection"""
        
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name="expense_documents")
            self.collection = self.client.get_or_create_collection(
                name="expense_documents",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("Vector store collection cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
