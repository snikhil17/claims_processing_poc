"""
FastAPI backend for Claims Processing AI
Two endpoints only: /health and /upload_test
Complete processing in one call
"""

import os
import shutil
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import core processing modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core.document_extractor import DocumentExtractor
from core.document_manager import DocumentManager  
from core.vector_store import VectorStore
from core.memory_utils import clear_memory, get_memory_stats
from config import config

# Langchain imports for LLM
from langchain_ollama import ChatOllama
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================================
# ENHANCED TASK MANAGER (FROM OLD APP)
# ================================

class TaskManager:
    """Enhanced task manager with all tasks from app_old.py"""
    
    def __init__(self):
        self.tasks = {
            "extract_amount": {
                "query": "total amount due payment cost price sum money dollar grand total balance final",
                "description": """Extract the FINAL total amount from this expense document. 
                Look for: 'Total', 'Amount Due', 'Grand Total', 'Balance Due' - usually at the bottom.
                Include currency. Don't confuse item prices with total amount.
                If you see subtotal + tax, add them for total.""",
                "prompt": """You are an expert expense analyst specialized in financial documents.

STEP-BY-STEP APPROACH:
1. Scan the context for ALL monetary values
2. Identify which values are totals vs individual items
3. Look for keywords: "Total", "Amount Due", "Grand Total", "Balance Due"
4. If you see "Subtotal" and "Tax", add them to get the total
5. The total amount is usually at the BOTTOM of the document

DOCUMENT CONTEXT:
{context}

Based on the above context, what is the TOTAL AMOUNT?
- Include currency symbol if shown
- If multiple totals exist, choose the final/grand total
- If not clearly found, respond "Information not found"

Total Amount:"""
            },
            
            "extract_date": {
                "query": "date transaction purchase invoice receipt timestamp when dated issued bill",
                "description": """Extract the main transaction/invoice date.
                Look for: 'Date:', 'Invoice Date:', 'Transaction Date:', 'Dated:'.
                Convert to YYYY-MM-DD format. Choose transaction date over print date.""",
                "prompt": """You are an expert document analyst.

DATE IDENTIFICATION STEPS:
1. Look for date labels: "Date:", "Invoice Date:", "Transaction Date:", "Dated:"
2. Identify all dates in the context
3. Prioritize transaction/invoice dates over other dates
4. Convert to YYYY-MM-DD format

DOCUMENT CONTEXT:
{context}

Extract the main transaction/invoice date in YYYY-MM-DD format:"""
            },
            
            "extract_vendor": {
                "query": "vendor merchant company business supplier store restaurant hotel from sold by provider",
                "description": """Extract the vendor/merchant name who provided the service.
                Look at the document header first. Check for business names in large/bold text.
                The vendor is who PROVIDED the service, not who received it.""",
                "prompt": """You are an expert at identifying business information.

VENDOR IDENTIFICATION PROCESS:
1. Check the TOP/HEADER of the document first
2. Look for business names in prominent positions
3. The vendor PROVIDES the service (not receives it)
4. Common patterns: company logos, "From:", "Merchant:", business registration numbers

DOCUMENT CONTEXT:
{context}

Who is the vendor/merchant? Extract the exact business name:"""
            },
            
            "extract_category": {
                "query": "category type classification expense kind service product item purchase",
                "description": "Determine expense category from this document based on the vendor and items",
                "prompt": """You are an expense categorization expert.

CATEGORIZATION STEPS:
1. Identify the vendor/merchant type
2. Look at the items or services provided
3. Common categories: Travel, Food & Dining, Accommodation, Transportation, Office Supplies, etc.

DOCUMENT CONTEXT:
{context}

What expense category does this document belong to?"""
            },
            
            "extract_items": {
                "query": "items products services line items purchases description details quantity price list",
                "description": "Extract ALL itemized details including descriptions and individual amounts. Look for tables or lists.",
                "prompt": """You are an itemization expert.

ITEM EXTRACTION STEPS:
1. Look for line items, products, or services
2. Include descriptions and individual prices
3. Look for tables or itemized lists
4. Include quantities if shown

DOCUMENT CONTEXT:
{context}

Extract all items/services with their details:"""
            },
            
            "extract_tax": {
                "query": "tax VAT GST sales tax rate percentage subtotal before tax",
                "description": "Extract tax amount AND tax rate if shown. Look for subtotal vs total difference.",
                "prompt": """You are a tax calculation expert.

TAX EXTRACTION STEPS:
1. Look for "Tax", "VAT", "GST", "Sales Tax"
2. Find tax amounts (usually with % or currency)
3. Check if there's a subtotal and total (difference might be tax)
4. Extract both tax amount AND rate if available

DOCUMENT CONTEXT:
{context}

Extract tax information (amount and rate if shown):"""
            }
        }
        
    def get_all_tasks(self) -> List[str]:
        """Get all available task names"""
        return list(self.tasks.keys())
    
    def get_task_info(self, task_name: str) -> Dict[str, str]:
        """Get task information"""
        return self.tasks.get(task_name, {})

# ================================
# CLAIMS PROCESSING AI
# ================================

class ClaimsProcessingAI:
    """Claims Processing AI for complete document processing"""
    
    def __init__(self):
        self.llm = ChatOllama(
            model="phi4-mini:3.8b",
            temperature=0.1,
            base_url="http://127.0.0.1:11434"
        )
        
        self.document_extractor = DocumentExtractor()
        self.document_manager = DocumentManager("data")
        self.vector_store = VectorStore("nomic-embed-text", "data/chroma_db")
        self.task_manager = TaskManager()
        
        logger.info("Claims Processing AI initialized")
    
    def process_document_complete(self, file_path: str) -> Dict[str, Any]:
        """Complete processing: extract, ingest, and run all tasks"""
        
        logger.info(f"Starting complete processing for: {file_path}")
        filename = Path(file_path).name
        
        try:
            # Step 1: Extract text
            logger.info("Step 1: Extracting text...")
            logger.info(f"Current working directory: {Path.cwd()}")
            logger.info(f"File path: {file_path}")
            logger.info(f"File exists: {Path(file_path).exists()}")
            
            raw_text = self.document_extractor.extract_text(file_path)
            logger.info(f"Raw text result: {type(raw_text)}, length: {len(raw_text) if raw_text else 0}")
            
            if not raw_text or len(raw_text.strip()) < 30:
                logger.error(f"Text extraction failed - raw_text: {raw_text is not None}, length: {len(raw_text.strip()) if raw_text else 0}")
                return {
                    "status": "error",
                    "message": "Failed to extract sufficient text from document",
                    "filename": filename
                }
            
            logger.info(f"Extracted {len(raw_text)} characters")
            
            # Step 2: Register document and create chunks
            logger.info("Step 2: Creating document chunks...")
            doc_id = self.document_manager.register_document(file_path, raw_text)
            document = self.document_manager.get_document(doc_id)
            
            # Step 3: Add to vector store
            logger.info("Step 3: Creating embeddings and storing in vector database...")
            self.vector_store.add_document_chunks(
                document_id=doc_id,
                chunks=document.chunks,
                metadata=document.metadata
            )
            
            # Step 4: Process all tasks
            logger.info("Step 4: Processing all extraction tasks...")
            task_results = {}
            
            for task_name in self.task_manager.get_all_tasks():
                logger.info(f"Processing task: {task_name}")
                
                task_result = self.process_single_task(doc_id, task_name)
                task_results[task_name] = task_result
            
            # Step 5: Compile final results
            logger.info("Step 5: Compiling results...")
            
            final_result = {
                "status": "success",
                "filename": filename,
                "document_id": doc_id,
                "text_length": len(raw_text),
                "chunks_created": len(document.chunks),
                "processing_timestamp": datetime.now().isoformat(),
                "extraction_results": task_results,
                "summary": {
                    "total_tasks": len(task_results),
                    "successful_tasks": sum(1 for r in task_results.values() if r.get("status") == "success"),
                    "failed_tasks": sum(1 for r in task_results.values() if r.get("status") == "error")
                }
            }
            
            # Clear memory after processing
            clear_memory()
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in complete processing: {e}")
            clear_memory()
            return {
                "status": "error",
                "message": str(e),
                "filename": filename
            }
    
    def process_single_task(self, document_id: str, task_name: str) -> Dict[str, Any]:
        """Process a single extraction task"""
        
        try:
            # Get task information
            task_info = self.task_manager.get_task_info(task_name)
            if not task_info:
                return {"status": "error", "message": f"Unknown task: {task_name}"}
            
            task_query = task_info.get("query", "")
            task_prompt = task_info.get("prompt", "")
            
            # Retrieve relevant chunks
            retrieval_results = self.vector_store.query_document(
                query=task_query,
                document_id=document_id,
                n_results=3
            )
            
            # Build context from retrieved chunks
            if retrieval_results.get("documents"):
                context = "\n\n---\n\n".join(retrieval_results["documents"])
            else:
                context = "No relevant context found"
            
            # Generate response using LLM
            prompt = task_prompt.format(context=context)
            response = self.llm.invoke(prompt)
            
            # Extract response content
            if hasattr(response, 'content'):
                response_text = response.content.strip()
            else:
                response_text = str(response).strip()
            
            return {
                "status": "success",
                "task": task_name,
                "response": response_text,
                "chunks_used": len(retrieval_results.get("documents", []))
            }
            
        except Exception as e:
            logger.error(f"Error processing task {task_name}: {e}")
            return {
                "status": "error",
                "task": task_name,
                "message": str(e)
            }

# Initialize FastAPI app
app = FastAPI(
    title="Claims Processing AI",
    description="Two endpoints: health check and complete document processing",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor
processor = ClaimsProcessingAI()

# Response models
class HealthResponse(BaseModel):
    status: str
    message: str
    ollama_running: bool
    memory_stats: Dict[str, Any]

class ProcessingResponse(BaseModel):
    status: str
    message: Optional[str] = None
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

# ================================
# CSV SAVING HELPER
# ================================

def _save_results_csv(csv_file_path: Path, complete_results: Dict[str, Any]) -> None:
    """
    Save processing results to CSV format for API
    
    Args:
        csv_file_path: Path to save the CSV file
        complete_results: Complete results dictionary
    """
    
    # Define CSV columns
    columns = [
        'filename', 'status', 'processing_timestamp',
        'expense_amount', 'expense_amount_confidence', 'expense_amount_currency',
        'vendor_name', 'vendor_name_confidence',
        'expense_date', 'expense_date_confidence',
        'expense_category', 'expense_category_confidence',
        'tax_amount', 'tax_amount_confidence', 'tax_rate', 'tax_type',
        'line_items', 'document_summary', 'document_summary_confidence',
        'error_message'
    ]
    
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        
        # Process each result
        for result in complete_results.get("results", []):
            row = {
                'filename': result.get('filename', ''),
                'processing_timestamp': complete_results.get("timestamp", ""),
                'status': result.get('status', ''),
                'error_message': result.get('message', '') if result.get('status') == 'error' else ''
            }
            
            # Extract task results if successful
            if result.get('status') == 'success' and 'extraction_results' in result:
                extraction_results = result['extraction_results']
                
                # Process expense_amount
                if 'expense_amount' in extraction_results and extraction_results['expense_amount'].get('status') == 'success':
                    response = extraction_results['expense_amount'].get('response', '')
                    try:
                        amount_data = json.loads(response)
                        row['expense_amount'] = amount_data.get('amount', '')
                        row['expense_amount_confidence'] = amount_data.get('confidence', '')
                        row['expense_amount_currency'] = amount_data.get('currency', '')
                    except (json.JSONDecodeError, KeyError):
                        row['expense_amount'] = response
                
                # Process vendor_name
                if 'vendor_name' in extraction_results and extraction_results['vendor_name'].get('status') == 'success':
                    response = extraction_results['vendor_name'].get('response', '')
                    try:
                        vendor_data = json.loads(response)
                        row['vendor_name'] = vendor_data.get('vendor_name', '')
                        row['vendor_name_confidence'] = vendor_data.get('confidence', '')
                    except (json.JSONDecodeError, KeyError):
                        row['vendor_name'] = response
                
                # Process expense_date
                if 'expense_date' in extraction_results and extraction_results['expense_date'].get('status') == 'success':
                    response = extraction_results['expense_date'].get('response', '')
                    try:
                        date_data = json.loads(response)
                        row['expense_date'] = date_data.get('expense_date', '')
                        row['expense_date_confidence'] = date_data.get('confidence', '')
                    except (json.JSONDecodeError, KeyError):
                        row['expense_date'] = response
                
                # Process expense_category
                if 'expense_category' in extraction_results and extraction_results['expense_category'].get('status') == 'success':
                    response = extraction_results['expense_category'].get('response', '')
                    try:
                        category_data = json.loads(response)
                        row['expense_category'] = category_data.get('category', '')
                        row['expense_category_confidence'] = category_data.get('confidence', '')
                    except (json.JSONDecodeError, KeyError):
                        row['expense_category'] = response
                
                # Process tax_amount
                if 'tax_amount' in extraction_results and extraction_results['tax_amount'].get('status') == 'success':
                    response = extraction_results['tax_amount'].get('response', '')
                    try:
                        tax_data = json.loads(response)
                        row['tax_amount'] = tax_data.get('tax_amount', '')
                        row['tax_amount_confidence'] = tax_data.get('confidence', '')
                        row['tax_rate'] = tax_data.get('tax_rate', '')
                        row['tax_type'] = tax_data.get('tax_type', '')
                    except (json.JSONDecodeError, KeyError):
                        row['tax_amount'] = response
                
                # Process line_items
                if 'line_items' in extraction_results and extraction_results['line_items'].get('status') == 'success':
                    response = extraction_results['line_items'].get('response', '')
                    try:
                        items_data = json.loads(response)
                        if isinstance(items_data, dict) and 'line_items' in items_data:
                            row['line_items'] = json.dumps(items_data['line_items'])
                        else:
                            row['line_items'] = response
                    except (json.JSONDecodeError, KeyError):
                        row['line_items'] = response
                
                # Process document_summary
                if 'document_summary' in extraction_results and extraction_results['document_summary'].get('status') == 'success':
                    response = extraction_results['document_summary'].get('response', '')
                    try:
                        summary_data = json.loads(response)
                        row['document_summary'] = summary_data.get('summary', '')
                        row['document_summary_confidence'] = summary_data.get('confidence', '')
                    except (json.JSONDecodeError, KeyError):
                        row['document_summary'] = response
            
            writer.writerow(row)

# ================================
# API ENDPOINTS (ONLY 2)
# ================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - verify system status"""
    
    logger.info("Health check requested")
    
    # Check Ollama status
    ollama_running = False
    try:
        import ollama
        ollama.list()
        ollama_running = True
        logger.info("Ollama is running")
    except Exception as e:
        logger.warning(f"Ollama check failed: {e}")
    
    # Get memory stats
    memory_stats = get_memory_stats()
    
    return HealthResponse(
        status="healthy",
        message="Claims Processing AI is running",
        ollama_running=ollama_running,
        memory_stats=memory_stats
    )

@app.post("/upload_test", response_model=ProcessingResponse)
async def upload_and_process_complete(
    files: List[UploadFile] = File(...)
):
    """
    Complete processing endpoint: Upload + Extract + Process All Tasks
    
    Accepts one or multiple documents and processes them completely:
    1. Upload and save files
    2. Extract text using OCR
    3. Create document chunks and embeddings
    4. Run all extraction tasks (amount, date, vendor, category, items, tax)
    5. Return complete results
    """
    
    logger.info(f"Upload and complete processing requested for {len(files)} files")
    
    # Validate files
    allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff']
    processed_results = []
    
    # Create upload directory
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    for file in files:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            processed_results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"File type not supported. Allowed: {allowed_extensions}"
            })
            continue
        
        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024
        if file.size > max_size:
            processed_results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"File too large. Maximum size: 10 MB"
            })
            continue
        
        try:
            # Save uploaded file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{file.filename}"
            file_path = upload_dir / filename
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"File saved: {file_path}")
            
            # Process document completely using absolute path
            absolute_file_path = file_path.resolve()
            result = processor.process_document_complete(str(absolute_file_path))
            processed_results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            processed_results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
    
    # Compile summary
    total_files = len(processed_results)
    successful_files = sum(1 for r in processed_results if r.get("status") == "success")
    failed_files = total_files - successful_files
    
    # Save results to output directory
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"processing_results_{timestamp}.json"
    csv_file = output_dir / f"processing_results_{timestamp}.csv"
    
    complete_results = {
        "timestamp": timestamp,
        "summary": {
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files
        },
        "results": processed_results
    }
    
    # Save JSON
    with open(output_file, 'w') as f:
        json.dump(complete_results, f, indent=2, default=str)
    
    # Save CSV
    _save_results_csv(csv_file, complete_results)
    
    logger.info(f"Results saved to: {output_file} and {csv_file}")
    
    return ProcessingResponse(
        status="success" if successful_files > 0 else "error",
        message=f"Processed {total_files} files: {successful_files} successful, {failed_files} failed",
        results=processed_results,
        summary={
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "output_file": str(output_file)
        }
    )

# ================================
# STARTUP/SHUTDOWN EVENTS
# ================================

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Starting Claims Processing AI API")
    
    # Ensure directories exist
    Path("data").mkdir(exist_ok=True)
    Path("data/uploads").mkdir(parents=True, exist_ok=True)
    Path("data/output").mkdir(parents=True, exist_ok=True)
    Path("data/documents").mkdir(parents=True, exist_ok=True)
    
    # Check Ollama connection
    try:
        import ollama
        ollama.list()
        logger.info("Ollama connection verified")
    except Exception as e:
        logger.error(f"Ollama connection failed: {e}")
        logger.warning("Make sure Ollama server is running: ollama serve")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Claims Processing AI API")
    clear_memory()

# ================================
# MAIN ENTRY POINT
# ================================

if __name__ == "__main__":
    # Get configuration
    host = "0.0.0.0"
    port = 8002  # Changed port again to avoid conflict
    reload = False
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"API documentation available at: http://{host}:{port}/docs")
    logger.info("Available endpoints:")
    logger.info("  GET /health - Health check")
    logger.info("  POST /upload_test - Complete document processing")
    
    # Run server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )