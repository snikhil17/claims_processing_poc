"""
Document management for Claims Processing AI with contextual chunking
"""

import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from .memory_utils import clear_cpu_memory, force_memory_cleanup

logger = logging.getLogger(__name__)

@dataclass 
class ClaimDocument:
    """Document with filename-based identification"""
    filename: str
    file_path: str
    raw_text: str
    chunks: List[str]
    metadata: Dict[str, Any]
    processed_timestamp: datetime

class FilenameBasedDocumentManager:
    """Manages documents using filenames as primary identifiers with contextual chunking"""
    
    def __init__(self):
        self.documents_registry = {}  # filename -> ClaimDocument
        self.chunk_to_file_map = {}  # chunk_id -> filename
        # Initialize LLM for contextual chunking
        try:
            from langchain_ollama import ChatOllama
            self.context_llm = ChatOllama(
                model="gemma3:4b",
                temperature=0.1,
                base_url="http://127.0.0.1:11434"
            )
        except Exception as e:
            print(f"⚠️ Could not initialize context LLM: {e}")
            self.context_llm = None
        
    def register_document(self, file_path: str, raw_text: str) -> str:
        """Register document using filename as ID with contextual chunking"""
        
        filename = Path(file_path).stem  # Get filename without extension
        
        print(f"🔄 Creating contextual chunks for {filename}...")
        
        # Create raw chunks first (smaller size for better context)
        raw_chunks = self.create_document_chunks(raw_text, filename)
        
        # Generate contextual chunks using LLM
        contextual_chunks = self.generate_contextual_chunks(raw_chunks, raw_text, filename)
        
        claim_doc = ClaimDocument(
            filename=filename,
            file_path=file_path,
            raw_text=raw_text,
            chunks=contextual_chunks,  # Use contextual chunks
            metadata={
                "file_name": Path(file_path).name,
                "file_extension": Path(file_path).suffix,
                "chunk_count": len(contextual_chunks),
                "source": "ocr_extraction",
                "chunking_method": "contextual",  # NEW metadata
                "raw_chunk_count": len(raw_chunks)
            },
            processed_timestamp=datetime.now()
        )
        
        self.documents_registry[filename] = claim_doc
        
        # Update chunk mapping
        for i, chunk in enumerate(contextual_chunks):
            chunk_id = f"{filename}_chunk_{i}"
            self.chunk_to_file_map[chunk_id] = filename
        
        print(f"✅ Created {len(contextual_chunks)} contextual chunks for {filename}")
        
        return filename
    
    def get_document_context(self, filename: str) -> ClaimDocument:
        """Get document context by filename"""
        if filename not in self.documents_registry:
            raise ValueError(f"Document {filename} not found in registry")
        return self.documents_registry[filename]
    
    def create_document_chunks(self, text: str, filename: str) -> List[str]:
        """Create smaller chunks for contextual chunking (reduced size for better context generation)"""
        
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_size = 300  # REDUCED from 500 for contextual chunking
        overlap_size = 50     # NEW: Small overlap for better context
        
        # Expense document section markers
        section_markers = [
            'total', 'amount', 'date', 'vendor', 'receipt', 'invoice',
            'item', 'quantity', 'price', 'tax', 'subtotal'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            line_length = len(line)
            is_section_start = any(marker in line.lower() for marker in section_markers)
            
            if (current_length + line_length > max_chunk_size) or \
               (is_section_start and current_chunk and current_length > 150):  # REDUCED threshold
                
                chunk_text = '\n'.join(current_chunk)
                if chunk_text.strip():
                    # Store raw chunk without document prefix for contextual processing
                    chunks.append(chunk_text)
                
                # Handle overlap for better context continuity
                if overlap_size > 0 and current_chunk:
                    overlap_lines = current_chunk[-2:]  # Keep last 2 lines for overlap
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
    
    def generate_contextual_chunks(self, raw_chunks: List[str], whole_document: str, filename: str) -> List[str]:
        """Generate contextual chunks using Anthropic's approach with smart cleanup scheduling"""
        
        if not self.context_llm:
            print("⚠️ Context LLM not available, using raw chunks")
            return [f"[DOCUMENT: {filename}]\n{chunk}" for chunk in raw_chunks]
        
        contextual_chunks = []
        
        # Anthropic's contextual retrieval prompt
        context_prompt_template = """<document>
{whole_document}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
        
        for i, chunk in enumerate(raw_chunks):
            try:
                # SMART CLEANUP: Clear memory every 3 chunks during contextual generation
                if i > 0 and i % 3 == 0:
                    print(f"🧹 Memory cleanup after chunk {i} (every 3 chunks)")
                    clear_cpu_memory()
                
                # Generate contextual prefix using LLM
                prompt = context_prompt_template.format(
                    whole_document=whole_document,
                    chunk_content=chunk
                )
                
                response = self.context_llm.invoke(prompt)
                
                # Safely extract content
                if hasattr(response, 'content'):
                    context_prefix = str(response.content).strip()
                else:
                    context_prefix = str(response).strip()
                
                # Clean up response object immediately
                del response
                
                # Ensure context is concise (50-100 tokens as per Anthropic)
                if len(context_prefix.split()) > 100:
                    context_prefix = ' '.join(context_prefix.split()[:100])
                
                # Create contextual chunk: context + original chunk
                contextual_chunk = f"Context: {context_prefix}\n\n{chunk}"
                contextual_chunks.append(contextual_chunk)
                
                print(f"✅ Generated context for chunk {i+1}/{len(raw_chunks)}")
                
                # Clean up variables to free memory
                del prompt, context_prefix, contextual_chunk
                
            except Exception as e:
                print(f"⚠️ Error generating context for chunk {i+1}: {e}")
                # SMART CLEANUP: Immediate aggressive cleanup on errors
                print("🧹 Immediate cleanup due to error")
                force_memory_cleanup()
                # Fallback to original chunk with document prefix
                contextual_chunks.append(f"[DOCUMENT: {filename}]\n{chunk}")
        
        # SMART CLEANUP: Full cleanup after all chunks processed
        print("🧹 Final cleanup after contextual chunk generation")
        force_memory_cleanup()
        
        return contextual_chunks

class DocumentManager(FilenameBasedDocumentManager):
    """Legacy DocumentManager class for backward compatibility"""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize document manager"""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.documents_dir = self.data_dir / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DocumentManager initialized with data_dir: {data_dir}")
    
    def generate_document_id(self, file_path: str, content: str) -> str:
        """Generate a unique document ID based on file path and content"""
        
        # Create hash from file path and content
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
        file_name = Path(file_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        return f"{file_name}_{timestamp}_{content_hash}"
    
    def create_chunks(self, content: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        """Split content into overlapping chunks"""
        
        if not content or len(content.strip()) < chunk_size:
            return [content] if content else []
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            
            # Ensure we don't split words
            if end < len(content) and not content[end].isspace():
                # Find the last space before the cutoff
                last_space = chunk.rfind(' ')
                if last_space > chunk_size // 2:  # Only if we don't go too far back
                    end = start + last_space
                    chunk = content[start:end]
            
            chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - overlap
            
            if start >= len(content):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def register_document(self, file_path: str, raw_text: str) -> str:
        """Register a new document and create chunks"""
        
        # Use the contextual chunking approach
        return super().register_document(file_path, raw_text)
    
    def get_document(self, doc_id: str) -> Optional[ClaimDocument]:
        """Get document by ID"""
        
        # For DocumentManager, doc_id is the filename directly (no splitting needed)
        try:
            return self.get_document_context(doc_id)
        except ValueError:
            return None
    
    def list_documents(self) -> List[str]:
        """List all document IDs"""
        
        return list(self.documents_registry.keys())
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document"""
        
        try:
            # Extract filename from doc_id
            filename = doc_id.split('_')[0] if '_' in doc_id else doc_id
            
            if filename in self.documents_registry:
                del self.documents_registry[filename]
            
            # Remove from chunk mapping
            chunk_ids_to_remove = [cid for cid, fname in self.chunk_to_file_map.items() if fname == filename]
            for cid in chunk_ids_to_remove:
                del self.chunk_to_file_map[cid]
            
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about managed documents"""
        
        all_docs = self.list_documents()
        total_chunks = 0
        total_content_length = 0
        
        for doc_id in all_docs:
            doc = self.get_document(doc_id)
            if doc:
                total_chunks += len(doc.chunks)
                total_content_length += len(doc.raw_text)
        
        return {
            "total_documents": len(all_docs),
            "total_chunks": total_chunks,
            "total_content_length": total_content_length,
            "average_chunks_per_doc": total_chunks / len(all_docs) if all_docs else 0,
            "documents_in_memory": len(self.documents_registry)
        }
    
    def _save_document(self, document: ClaimDocument):
        """Save document to disk"""
        
        try:
            doc_file = self.documents_dir / f"{document.filename}.json"
            
            # Convert document to dict for JSON serialization
            doc_dict = {
                "filename": document.filename,
                "file_path": document.file_path,
                "raw_text": document.raw_text,
                "chunks": document.chunks,
                "metadata": document.metadata,
                "processed_timestamp": document.processed_timestamp.isoformat()
            }
            
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(doc_dict, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving document {document.filename}: {e}")
    
    def _load_document(self, doc_id: str) -> Optional[ClaimDocument]:
        """Load document from disk"""
        
        try:
            filename = doc_id.split('_')[0] if '_' in doc_id else doc_id
            doc_file = self.documents_dir / f"{filename}.json"
            
            if not doc_file.exists():
                return None
            
            with open(doc_file, 'r', encoding='utf-8') as f:
                doc_dict = json.load(f)
            
            # Convert back to ClaimDocument object
            document = ClaimDocument(
                filename=doc_dict["filename"],
                file_path=doc_dict["file_path"],
                raw_text=doc_dict["raw_text"],
                chunks=doc_dict["chunks"],
                metadata=doc_dict["metadata"],
                processed_timestamp=datetime.fromisoformat(doc_dict["processed_timestamp"])
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Error loading document {doc_id}: {e}")
            return None