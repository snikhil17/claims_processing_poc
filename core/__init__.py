# ===== core/__init__.py =====
"""
Core processing modules for Claims Processing AI
"""

from .rag_processor import ClaimsProcessingAI, ProcessingPipeline
from .document_extractor import DocumentExtractor
from .document_manager import DocumentManager
from .vector_store import VectorStore
from .task_manager import TaskManager
from .memory_utils import clear_memory, get_memory_stats

__all__ = [
    'ClaimsProcessingAI',
    'ProcessingPipeline',
    'DocumentExtractor',
    'DocumentManager',
    'VectorStore',
    'TaskManager',
    'clear_memory',
    'get_memory_stats'
]

# ===== core/document_extractor.py =====
"""
Document extraction using IBM Granite Docling
"""

from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DocumentExtractor:
    """Extract text from documents using IBM Granite Docling"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']
        self._init_docling()
    
    def _init_docling(self):
        """Initialize Docling converter"""
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.do_table_structure = True
            
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            self.has_docling = True
            logger.info("Docling converter initialized")
            
        except Exception as e:
            self.has_docling = False
            logger.error(f"Failed to initialize Docling: {e}")
    
    def extract_text(self, file_path: str) -> Optional[str]:
        """Extract text from document"""
        
        if not self.has_docling:
            logger.error("Docling not available")
            return self._fallback_extraction(file_path)
        
        try:
            result = self.converter.convert(file_path)
            if result.document:
                text = result.document.export_to_markdown()
                return self._clean_text(text)
            return ""
            
        except Exception as e:
            logger.error(f"Docling extraction failed: {e}")
            return self._fallback_extraction(file_path)
    
    def _fallback_extraction(self, file_path: str) -> str:
        """Fallback extraction using pytesseract"""
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return self._clean_text(text)
            
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        if not text:
            return ""
        
        lines = text.split('\n')
        cleaned = []
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:
                cleaned.append(line)
        
        return '\n'.join(cleaned)

# ===== core/document_manager.py =====
"""
Document and chunk management
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document with metadata and chunks"""
    document_id: str
    file_path: str
    raw_text: str
    chunks: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime

class DocumentManager:
    """Manage documents and chunking"""
    
    def __init__(self):
        self.documents = {}  # document_id -> Document
        self.chunk_size = 300
        self.chunk_overlap = 50
    
    def register_document(self, file_path: str, raw_text: str) -> str:
        """Register document and create chunks"""
        
        # Generate document ID from filename
        doc_id = Path(file_path).stem
        
        # Create chunks
        chunks = self._create_chunks(raw_text)
        
        # Create document object
        document = Document(
            document_id=doc_id,
            file_path=file_path,
            raw_text=raw_text,
            chunks=chunks,
            metadata={
                "filename": Path(file_path).name,
                "extension": Path(file_path).suffix,
                "chunk_count": len(chunks),
                "text_length": len(raw_text)
            },
            timestamp=datetime.now()
        )
        
        self.documents[doc_id] = document
        logger.info(f"Registered document: {doc_id} with {len(chunks)} chunks")
        
        return doc_id
    
    def get_document(self, doc_id: str) -> Document:
        """Get document by ID"""
        if doc_id not in self.documents:
            raise ValueError(f"Document {doc_id} not found")
        return self.documents[doc_id]
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create text chunks with overlap"""
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Keywords that indicate section boundaries
        section_markers = [
            'total', 'amount', 'date', 'vendor', 'receipt',
            'invoice', 'item', 'price', 'tax', 'subtotal'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            line_length = len(line)
            is_section = any(marker in line.lower() for marker in section_markers)
            
            # Check if we should start a new chunk
            if (current_length + line_length > self.chunk_size) or \
               (is_section and current_chunk and current_length > 150):
                
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                if chunk_text.strip():
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_lines = current_chunk[-2:]
                    current_chunk = overlap_lines + [line]
                    current_length = sum(len(l) for l in overlap_lines) + line_length
                else:
                    current_chunk = [line]
                    current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length + 1
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks

# ===== core/vector_store.py =====
"""
Vector storage using ChromaDB
"""

from typing import List, Dict, Any
import chromadb
import ollama
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """ChromaDB vector storage with document isolation"""
    
    def __init__(self, embedding_model: str = "nomic-embed-text"):
        self.embedding_model = embedding_model
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./data/chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="expense_documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info("Vector store initialized")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using Ollama"""
        try:
            response = ollama.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return []
    
    def add_document_chunks(self, document_id: str, chunks: List[str], 
                          metadata: Dict[str, Any]):
        """Add document chunks to vector store"""
        
        embeddings = []
        chunk_ids = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            embedding = self.embed_text(chunk)
            if not embedding:
                continue
            
            chunk_id = f"{document_id}_chunk_{i}"
            chunk_metadata = {
                **metadata,
                "document_id": document_id,
                "chunk_index": i
            }
            
            embeddings.append(embedding)
            chunk_ids.append(chunk_id)
            metadatas.append(chunk_metadata)
        
        # Add to ChromaDB
        if embeddings:
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=chunk_ids
            )
            logger.info(f"Added {len(embeddings)} chunks for {document_id}")
    
    def query_document(self, query: str, document_id: str, 
                      n_results: int = 3) -> Dict[str, Any]:
        """Query specific document"""
        
        query_embedding = self.embed_text(query)
        if not query_embedding:
            return {"error": "Failed to generate query embedding"}
        
        # Query with document filter
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"document_id": document_id},
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            "documents": results['documents'][0] if results['documents'] else [],
            "metadatas": results['metadatas'][0] if results['metadatas'] else [],
            "distances": results['distances'][0] if results['distances'] else []
        }

# ===== core/task_manager.py =====
"""
Task definitions and management
"""

from typing import Dict, List, Any

class TaskManager:
    """Manage extraction tasks"""
    
    def __init__(self):
        self.tasks = {
            "extract_amount": {
                "query": "total amount due payment cost price sum money dollar grand total balance final",
                "description": "Extract the FINAL total amount from expense document",
                "format": "numeric value with currency"
            },
            "extract_date": {
                "query": "date transaction purchase invoice receipt timestamp when dated issued bill",
                "description": "Extract the main transaction/invoice date",
                "format": "YYYY-MM-DD format"
            },
            "extract_vendor": {
                "query": "vendor merchant company business supplier store restaurant hotel from sold by provider",
                "description": "Extract the vendor/merchant name",
                "format": "company or business name"
            },
            "extract_category": {
                "query": "category type classification expense kind service product item purchase",
                "description": "Determine expense category",
                "format": "expense category"
            },
            "extract_items": {
                "query": "items products services line items purchases description details quantity price list",
                "description": "Extract itemized details",
                "format": "list of items"
            },
            "extract_tax": {
                "query": "tax VAT GST sales tax rate percentage subtotal before tax",
                "description": "Extract tax amount and rate",
                "format": "tax amount and rate"
            }
        }
    
    def get_task_info(self, task_name: str) -> Dict[str, str]:
        """Get task information"""
        return self.tasks.get(task_name, {})
    
    def list_tasks(self) -> List[str]:
        """List available tasks"""
        return list(self.tasks.keys())
    
    def get_all_tasks(self) -> Dict[str, Dict[str, str]]:
        """Get all task definitions"""
        return self.tasks

# ===== core/memory_utils.py =====
"""
Memory management utilities
"""

import gc
import logging

logger = logging.getLogger(__name__)

def clear_memory():
    """Clear CPU memory"""
    try:
        import psutil
        import os
        
        # Force garbage collection
        gc.collect()
        
        # Get memory info
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Another garbage collection
        gc.collect()
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"Memory cleared: {memory_before:.1f} MB → {memory_after:.1f} MB")
        
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")

def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics"""
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {}

# ===== api/__init__.py =====
"""
FastAPI backend for RAG Expense Processor
"""

# ===== api/models.py =====
"""
Pydantic models for API requests and responses
"""

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    document_id: str
    filename: str
    chunks_created: int
    text_length: int
    timestamp: datetime

class TaskRequest(BaseModel):
    """Request for single task processing"""
    document_id: str
    task_name: str

class TaskResponse(BaseModel):
    """Response for task processing"""
    task: str
    response: str
    chunks_used: int
    confidence: Optional[float] = None

class BatchProcessRequest(BaseModel):
    """Request for batch processing"""
    document_ids: List[str]
    task_names: Optional[List[str]] = None

class ProcessingStatus(BaseModel):
    """Processing status response"""
    status: str
    progress: int
    current_task: Optional[str] = None
    results: Optional[Dict[str, Any]] = None