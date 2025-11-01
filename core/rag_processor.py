"""
Main Claims Processing AI logic for expense documents
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from langchain_ollama import ChatOllama
import chromadb
import ollama

from .document_extractor import DocumentExtractor
from .document_manager import DocumentManager
from .vector_store import VectorStore
from .task_manager import TaskManager
from .memory_utils import clear_memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaimsProcessingAI:
    """
    Main Claims Processing AI for expense documents
    Handles ingestion, retrieval, and task-based extraction
    """
    
    def __init__(self, text_model: str = "gemma3:4b", 
                 embedding_model: str = "nomic-embed-text"):
        """Initialize the Claims Processing AI"""
        
        self.text_model = text_model
        self.embedding_model = embedding_model
        
        # Initialize components
        self.llm = ChatOllama(
            model=text_model,
            temperature=0.1,
            base_url="http://127.0.0.1:11434"
        )
        
        self.document_extractor = DocumentExtractor()
        self.document_manager = DocumentManager()
        self.vector_store = VectorStore(embedding_model)
        self.task_manager = TaskManager()
        
        logger.info(f"Claims Processing AI initialized with {text_model}")
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a document: extract text, create chunks, generate embeddings
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict with ingestion results
        """
        
        logger.info(f"Ingesting document: {file_path}")
        
        try:
            # Extract text from document
            raw_text = self.document_extractor.extract_text(file_path)
            
            if not raw_text or len(raw_text.strip()) < 50:
                return {
                    "status": "error",
                    "message": "Failed to extract sufficient text from document"
                }
            
            # Register document and create chunks
            doc_id = self.document_manager.register_document(file_path, raw_text)
            document = self.document_manager.get_document(doc_id)
            
            # Add to vector store
            self.vector_store.add_document_chunks(
                document_id=doc_id,
                chunks=document.chunks,
                metadata=document.metadata
            )
            
            # Clear memory after ingestion
            clear_memory()
            
            return {
                "status": "success",
                "document_id": doc_id,
                "chunks_created": len(document.chunks),
                "text_length": len(raw_text)
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {e}")
            clear_memory()
            return {
                "status": "error",
                "message": str(e)
            }
    
    def process_task(self, document_id: str, task_name: str) -> Dict[str, Any]:
        """
        Process a specific extraction task for a document
        
        Args:
            document_id: ID of the document
            task_name: Name of the extraction task
            
        Returns:
            Dict with task results
        """
        
        logger.info(f"Processing task '{task_name}' for document '{document_id}'")
        
        try:
            # Get task information
            task_info = self.task_manager.get_task_info(task_name)
            if not task_info:
                return {
                    "status": "error",
                    "message": f"Unknown task: {task_name}"
                }
            
            # Retrieve relevant chunks
            retrieval_results = self.vector_store.query_document(
                query=task_info["query"],
                document_id=document_id,
                n_results=5
            )
            
            if not retrieval_results["documents"]:
                return {
                    "status": "error",
                    "message": "No relevant context found"
                }
            
            # Optimize context
            context = self._optimize_context(retrieval_results, task_name)
            
            # Generate response with LLM
            response = self._generate_task_response(
                context=context,
                task_name=task_name,
                task_description=task_info["description"]
            )
            
            # Clear memory after task
            clear_memory()
            
            return {
                "status": "success",
                "task": task_name,
                "response": response,
                "chunks_used": len(retrieval_results["documents"])
            }
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            clear_memory()
            return {
                "status": "error",
                "message": str(e)
            }
    
    def process_all_tasks(self, document_id: str) -> Dict[str, Any]:
        """
        Process all extraction tasks for a document
        
        Args:
            document_id: ID of the document
            
        Returns:
            Dict with all task results
        """
        
        logger.info(f"Processing all tasks for document '{document_id}'")
        
        results = {}
        tasks = self.task_manager.list_tasks()
        
        for task_name in tasks:
            result = self.process_task(document_id, task_name)
            results[task_name] = result
            
            # Clear memory between tasks
            if len(tasks) > 3:
                clear_memory()
        
        return results
    
    def _optimize_context(self, retrieval_results: Dict[str, Any], 
                         task_name: str) -> str:
        """
        Optimize retrieved context for the specific task
        
        Args:
            retrieval_results: Retrieved chunks and metadata
            task_name: Name of the task
            
        Returns:
            Optimized context string
        """
        
        documents = retrieval_results.get("documents", [])
        distances = retrieval_results.get("distances", [])
        
        if not documents:
            return ""
        
        # Score and rank documents
        scored_docs = []
        for doc, distance in zip(documents, distances):
            score = 1.0 / (1.0 + distance)
            
            # Boost score based on task-specific keywords
            doc_lower = doc.lower()
            if task_name == "extract_amount":
                if any(word in doc_lower for word in ["total", "amount due", "grand total"]):
                    score *= 1.5
            elif task_name == "extract_date":
                if any(word in doc_lower for word in ["date", "dated", "issued"]):
                    score *= 1.5
            elif task_name == "extract_vendor":
                if any(word in doc_lower for word in ["company", "corp", "inc", "llc"]):
                    score *= 1.3
            
            scored_docs.append((doc, score))
        
        # Sort by score and build context
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        optimized_chunks = []
        total_length = 0
        max_length = 2500
        
        for doc, score in scored_docs[:4]:
            if total_length + len(doc) <= max_length:
                optimized_chunks.append(f"[Relevance: {score:.2f}]\n{doc}")
                total_length += len(doc)
        
        return "\n\n---\n\n".join(optimized_chunks)
    
    def _generate_task_response(self, context: str, task_name: str, 
                               task_description: str) -> str:
        """
        Generate LLM response for the task
        
        Args:
            context: Optimized context
            task_name: Name of the task
            task_description: Task description
            
        Returns:
            LLM response
        """
        
        # Task-specific prompts
        prompts = {
            "extract_amount": """You are an expense analyst.

DOCUMENT CONTEXT:
{context}

Extract the TOTAL AMOUNT from this expense document.
Look for: 'Total', 'Amount Due', 'Grand Total', 'Balance Due'
Include currency symbol if shown.

Total Amount:""",
            
            "extract_date": """You are a document analyst.

DOCUMENT CONTEXT:
{context}

Extract the main transaction/invoice date.
Convert to YYYY-MM-DD format.

Date:""",
            
            "extract_vendor": """You are a document analyst.

DOCUMENT CONTEXT:
{context}

Extract the vendor/merchant name who provided the service.

Vendor:""",
            
            "extract_category": """You are an expense categorization expert.

DOCUMENT CONTEXT:
{context}

Determine the expense category (Travel, Food & Dining, Accommodation, etc.)

Category:""",
            
            "extract_tax": """You are a tax analyst.

DOCUMENT CONTEXT:
{context}

Extract tax amount and rate if shown.

Tax Information:""",
            
            "extract_items": """You are an itemization expert.

DOCUMENT CONTEXT:
{context}

Extract all items/services with their details.

Items:"""
        }
        
        # Use specific prompt or fallback
        if task_name in prompts:
            prompt = prompts[task_name].format(context=context)
        else:
            prompt = f"""{task_description}

DOCUMENT CONTEXT:
{context}

Response:"""
        
        try:
            response = self.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                return str(response.content).strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return f"Error: {e}"

class ProcessingPipeline:
    """
    High-level pipeline for processing multiple documents
    """
    
    def __init__(self):
        self.processor = ClaimsProcessingAI()
        self.results = {}
    
    def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple documents through the full pipeline
        
        Args:
            file_paths: List of document file paths
            
        Returns:
            Processing results
        """
        
        logger.info(f"Processing {len(file_paths)} documents")
        
        pipeline_results = {
            "timestamp": datetime.now().isoformat(),
            "total_documents": len(file_paths),
            "successful": 0,
            "failed": 0,
            "documents": {}
        }
        
        for file_path in file_paths:
            filename = Path(file_path).name
            logger.info(f"Processing: {filename}")
            
            # Ingest document
            ingestion_result = self.processor.ingest_document(file_path)
            
            if ingestion_result["status"] == "success":
                doc_id = ingestion_result["document_id"]
                
                # Process all tasks
                task_results = self.processor.process_all_tasks(doc_id)
                
                pipeline_results["documents"][filename] = {
                    "document_id": doc_id,
                    "ingestion": ingestion_result,
                    "tasks": task_results
                }
                pipeline_results["successful"] += 1
            else:
                pipeline_results["documents"][filename] = {
                    "error": ingestion_result.get("message", "Unknown error")
                }
                pipeline_results["failed"] += 1
            
            # Clear memory between documents
            clear_memory()
        
        self.results = pipeline_results
        return pipeline_results
    
    def save_results(self, output_dir: str = "data/output") -> str:
        """
        Save processing results to JSON and CSV
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Path to saved JSON file
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"processing_results_{timestamp}.json"
        csv_filename = f"processing_results_{timestamp}.csv"
        
        json_file_path = output_path / json_filename
        csv_file_path = output_path / csv_filename
        
        # Save JSON
        with open(json_file_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save CSV
        self._save_results_csv(csv_file_path)
        
        logger.info(f"Results saved to {json_file_path} and {csv_file_path}")
        return str(json_file_path)
    
    def _save_results_csv(self, csv_file_path: Path) -> None:
        """
        Save processing results to CSV format
        
        Args:
            csv_file_path: Path to save the CSV file
        """
        
        # Define CSV columns
        columns = [
            'filename', 'status', 'document_id', 'processing_timestamp',
            'expense_amount', 'expense_amount_confidence', 'expense_amount_currency',
            'vendor_name', 'vendor_name_confidence',
            'expense_date', 'expense_date_confidence',
            'expense_category', 'expense_category_confidence',
            'tax_amount', 'tax_amount_confidence', 'tax_rate', 'tax_type',
            'document_summary', 'document_summary_confidence',
            'error_message'
        ]
        
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            
            # Process each document
            for filename, doc_data in self.results.get("documents", {}).items():
                row = {
                    'filename': filename,
                    'processing_timestamp': self.results.get("timestamp", ""),
                    'status': 'success' if 'document_id' in doc_data else 'error',
                    'document_id': doc_data.get('document_id', ''),
                    'error_message': doc_data.get('error', '')
                }
                
                # Extract task results if successful
                if 'tasks' in doc_data:
                    tasks = doc_data['tasks']
                    
                    # Process expense_amount
                    if 'expense_amount' in tasks and tasks['expense_amount'].get('status') == 'success':
                        try:
                            amount_data = json.loads(tasks['expense_amount']['response'])
                            row['expense_amount'] = amount_data.get('amount', '')
                            row['expense_amount_confidence'] = amount_data.get('confidence', '')
                            row['expense_amount_currency'] = amount_data.get('currency', '')
                        except (json.JSONDecodeError, KeyError):
                            row['expense_amount'] = tasks['expense_amount'].get('response', '')
                    
                    # Process vendor_name
                    if 'vendor_name' in tasks and tasks['vendor_name'].get('status') == 'success':
                        try:
                            vendor_data = json.loads(tasks['vendor_name']['response'])
                            row['vendor_name'] = vendor_data.get('vendor_name', '')
                            row['vendor_name_confidence'] = vendor_data.get('confidence', '')
                        except (json.JSONDecodeError, KeyError):
                            row['vendor_name'] = tasks['vendor_name'].get('response', '')
                    
                    # Process expense_date
                    if 'expense_date' in tasks and tasks['expense_date'].get('status') == 'success':
                        try:
                            date_data = json.loads(tasks['expense_date']['response'])
                            row['expense_date'] = date_data.get('expense_date', '')
                            row['expense_date_confidence'] = date_data.get('confidence', '')
                        except (json.JSONDecodeError, KeyError):
                            row['expense_date'] = tasks['expense_date'].get('response', '')
                    
                    # Process expense_category
                    if 'expense_category' in tasks and tasks['expense_category'].get('status') == 'success':
                        try:
                            category_data = json.loads(tasks['expense_category']['response'])
                            row['expense_category'] = category_data.get('category', '')
                            row['expense_category_confidence'] = category_data.get('confidence', '')
                        except (json.JSONDecodeError, KeyError):
                            row['expense_category'] = tasks['expense_category'].get('response', '')
                    
                    # Process tax_amount
                    if 'tax_amount' in tasks and tasks['tax_amount'].get('status') == 'success':
                        try:
                            tax_data = json.loads(tasks['tax_amount']['response'])
                            row['tax_amount'] = tax_data.get('tax_amount', '')
                            row['tax_amount_confidence'] = tax_data.get('confidence', '')
                            row['tax_rate'] = tax_data.get('tax_rate', '')
                            row['tax_type'] = tax_data.get('tax_type', '')
                        except (json.JSONDecodeError, KeyError):
                            row['tax_amount'] = tasks['tax_amount'].get('response', '')
                    
                    # Process document_summary
                    if 'document_summary' in tasks and tasks['document_summary'].get('status') == 'success':
                        try:
                            summary_data = json.loads(tasks['document_summary']['response'])
                            row['document_summary'] = summary_data.get('summary', '')
                            row['document_summary_confidence'] = summary_data.get('confidence', '')
                        except (json.JSONDecodeError, KeyError):
                            row['document_summary'] = tasks['document_summary'].get('response', '')
                
                writer.writerow(row)