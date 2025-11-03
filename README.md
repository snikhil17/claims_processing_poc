# Claims Processing AI - Offline Document Processing System

A Claims Processing AI-based expense document processing system that works completely offline after initial setup.

## System Architecture

The system follows a modular architecture designed for offline operation:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  FastAPI    │────▶│     Core     │────▶│   ChromaDB  │
│   Backend   │     │  Processing  │     │   Vector    │
└─────────────┘     │    Engine    │     │    Store    │
                    └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    Ollama    │
                    │  LLM Models  │
                    └──────────────┘
```

## Key Features

- **Completely Offline Operation**: After initial setup, no internet required
- **IBM Granite Docling OCR**: State-of-the-art document extraction
- **Task-Based Extraction**: Specialized extraction for amounts, dates, vendors, etc.
- **RESTful API (FAstAPI)**: Easy integration with any client application
- **Memory Management**: Automatic memory optimization for long-running operations

## Installation & Setup

### Prerequisites

- Python 3.11.13 or higher
- 8GB+ RAM recommended
- 10GB+ free disk space for models (because downloading LLMs)

### Step 1: Initial Setup (Internet Required)

Run the setup script while connected to the internet. This will download all necessary models and dependencies:

```bash
# From project root
python setup.py
```

This will:
- Install all Python dependencies
- Download Ollama models (phi4-mini:3.8b and nomic-embed-text)
- Pre-download IBM Granite Docling models
- Initialize ChromaDB
- Create necessary directories
- Generate configuration file

### Step 2: Verify Setup (Offline) - Only to check and execute, in case of any error (related to ollama) in Step-1

After running setup.py, check the output for any Ollama-related issues:

```bash
# Check if Ollama is properly installed
ollama --version

# If Ollama is not installed, install it:
# Windows: Download from https://ollama.ai/download
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh

# Verify Ollama models were downloaded during setup
ollama list
# Should show: phi4-mini:3.8b and nomic-embed-text

# If models are missing, pull them manually
ollama pull phi4-mini:3.8b
ollama pull nomic-embed-text

# Start Ollama server in background
ollama serve
```

**Troubleshooting Ollama Issues:**
- If `ollama --version` fails: Reinstall Ollama from https://ollama.ai
- If models didn't download: Check internet connection during setup.py
- If `ollama serve` fails: Ensure port 11434 is not in use by another service

### Step 3: Run the API Server (Offline)

```bash
# From project root
python api/main.py
```

The API will be available at:
- API Endpoint: `http://localhost:8002`
- Interactive Docs: `http://localhost:8002/docs`
- ReDoc: `http://localhost:8002/redoc`

## API Usage

### 1. Interactive API Documentation (Swagger UI)

When the API server is running, access the interactive documentation at:
- **Swagger UI**: `http://localhost:8002/docs`
- **ReDoc**: `http://localhost:8002/redoc`

**Features:**
- ✅ **Interactive Testing**: Test all endpoints directly from the browser
- ✅ **Request/Response Examples**: See sample requests and responses
- ✅ **Schema Documentation**: View detailed parameter and response schemas
- ✅ **Try It Out**: Execute API calls with custom parameters
- ✅ **Authentication**: No auth required for this API

**How to use Swagger UI:**
1. Start the API server: `python api/main.py`
2. Open `http://localhost:8002/docs` in your browser
3. Click on an endpoint to expand it
4. Click "Try it out" to test the endpoint
5. Fill in parameters and click "Execute"

### 2. Check System Status

```bash
curl http://localhost:8002/health
```

### 3. Upload and Process Document

```bash
# Upload and process document (complete processing in one call)
# Replace 'path/to/your/document.pdf' with actual file path
curl -X POST "http://localhost:8002/upload_test" \
  -F "files=@path/to/your/document.pdf"
```

**Example with actual file:**
```bash
# First, navigate to the project directory
cd rag_expense_processor

# Using a sample receipt image
curl -X POST "http://localhost:8002/upload_test" \
  -F "files=@data/uploads/hotel-receipt-25.jpg"
```

**Note**: Make sure you're in the `rag_expense_processor` directory when running the curl command, or use the full path: `rag_expense_processor/data/uploads/hotel-receipt-25.jpg`

**PowerShell (Windows) alternative:**
```powershell
# First, navigate to the project directory
cd rag_expense_processor

# Using Python (recommended for Windows)
python -c "
import requests
files = {'files': open('data/uploads/hotel-receipt-25.jpg', 'rb')}
response = requests.post('http://localhost:8002/upload_test', files=files)
print(response.json())
"
```

### 4. Batch Process Multiple Documents

**Windows (PowerShell/Python):**
```powershell
# Navigate to project directory
cd rag_expense_processor

# Using Python for batch processing
python -c "
import requests

# List of files to process (update paths as needed)
file_paths = [
    'data/uploads/invoice1.pdf',
    'data/uploads/receipt2.jpg',
    'data/uploads/bill3.png'
]

files = [('files', open(file_path, 'rb')) for file_path in file_paths]

try:
    response = requests.post('http://localhost:8002/upload_test', files=files)
    result = response.json()
    
    print('Status:', result['status'])
    print('Message:', result['message'])
    print('Total files processed:', result['summary']['total_files'])
    print('Successful:', result['summary']['successful_files'])
    print('Failed:', result['summary']['failed_files'])
    
    # Show results for each document
    for doc_result in result['results']:
        print(f'\\nDocument: {doc_result[\"filename\"]}')
        if doc_result['status'] == 'success':
            print('✅ Processing successful')
            extraction = doc_result.get('extraction_results', {})
            if 'extract_amount' in extraction:
                print('Amount:', extraction['extract_amount'].get('response', 'N/A'))
            if 'extract_vendor' in extraction:
                print('Vendor:', extraction['extract_vendor'].get('response', 'N/A'))
        else:
            print('❌ Processing failed:', doc_result.get('message', 'Unknown error'))

finally:
    # Always close the files
    for _, file_obj in files:
        file_obj.close()
"
```

**WSL/Linux (Bash/Python):**
```bash
# Navigate to project directory
cd rag_expense_processor

# Using Python for batch processing
python3 -c "
import requests

# List of files to process (update paths as needed)
file_paths = [
    'data/uploads/invoice1.pdf',
    'data/uploads/receipt2.jpg',
    'data/uploads/bill3.png'
]

files = [('files', open(file_path, 'rb')) for file_path in file_paths]

try:
    response = requests.post('http://localhost:8002/upload_test', files=files)
    result = response.json()
    
    print('Status:', result['status'])
    print('Message:', result['message'])
    print('Total files processed:', result['summary']['total_files'])
    print('Successful:', result['summary']['successful_files'])
    print('Failed:', result['summary']['failed_files'])
    
    # Show results for each document
    for doc_result in result['results']:
        print(f'\\nDocument: {doc_result[\"filename\"]}')
        if doc_result['status'] == 'success':
            print('✅ Processing successful')
            extraction = doc_result.get('extraction_results', {})
            if 'extract_amount' in extraction:
                print('Amount:', extraction['extract_amount'].get('response', 'N/A'))
            if 'extract_vendor' in extraction:
                print('Vendor:', extraction['extract_vendor'].get('response', 'N/A'))
        else:
            print('❌ Processing failed:', doc_result.get('message', 'Unknown error'))

finally:
    # Always close the files
    for _, file_obj in files:
        file_obj.close()
"
```

### 5. Testing with Postman/Insomnia

**Import API Collection:**
1. Open Postman or Insomnia
2. Create a new collection called "Claims Processing AI"
3. Add requests for each endpoint:

**Health Check Request:**
- Method: GET
- URL: `http://localhost:8002/health`
- Headers: `Accept: application/json`

**Document Upload Request:**
- Method: POST
- URL: `http://localhost:8002/upload_test`
- Body: form-data
  - Key: `files`, Type: File, Select file(s) to upload

**Environment Variables:**
Set up environment variables for easy testing:
- `base_url`: `http://localhost:8002`
- `upload_file`: path to test document

### 6. Automated API Testing

**Basic Test Suite:**
```python
import requests
import json

def test_api_endpoints():
    """Comprehensive API testing suite"""
    base_url = "http://localhost:8002"
    
    print("🧪 Starting API Tests...")
    
    # Test 1: Health Check
    print("\\n1. Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "ollama_running" in data
        print("✅ Health check passed")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Document Processing
    print("\\n2. Testing Document Processing...")
    try:
        # Use a test file (adjust path as needed)
        test_file = "data/uploads/hotel-receipt-25.jpg"
        files = {'files': open(test_file, 'rb')}
        response = requests.post(f"{base_url}/upload_test", files=files)
        files['files'].close()
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] in ["success", "error"]
        assert "summary" in result
        assert "results" in result
        print("✅ Document processing test passed")
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False
    
    print("\\n🎉 All API tests passed!")
    return True

if __name__ == "__main__":
    test_api_endpoints()
```

### 7. Response Format Reference

**Health Endpoint Response:**
```json
{
  "status": "healthy",
  "message": "Claims Processing AI is running",
  "ollama_running": true,
  "memory_stats": {
    "process_memory_mb": 245.8,
    "process_memory_percent": 1.53,
    "system_memory_total_mb": 16086.93,
    "system_memory_available_mb": 2891.2,
    "system_memory_percent": 82.0,
    "cpu_percent": 5.2
  }
}
```

**Document Processing Response:**
```json
{
  "status": "success",
  "message": "Processed 1 files: 1 successful, 0 failed",
  "results": [
    {
      "status": "success",
      "filename": "hotel-receipt-25.jpg",
      "document_id": "doc_123",
      "text_length": 1250,
      "chunks_created": 4,
      "processing_timestamp": "2025-09-23T10:30:00",
      "extraction_results": {
        "extract_amount": {
          "status": "success",
          "response": "$125.50",
          "chunks_used": 2
        },
        "extract_vendor": {
          "status": "success", 
          "response": "Hotel California",
          "chunks_used": 1
        },
        "extract_date": {
          "status": "success",
          "response": "2025-09-20",
          "chunks_used": 1
        }
      }
    }
  ],
  "summary": {
    "total_files": 1,
    "successful_files": 1,
    "failed_files": 0,
    "output_file": "data/output/processing_results_20250923_103000.json"
  }
}
```

## Claims Processing AI Storage & Processing Flow

When you upload documents through the API, they go through several processing stages with data stored in specific folders:

### 📁 **Document Storage Locations**

**1. `data/uploads/` - Uploaded Documents**
- **Purpose**: Temporary storage for uploaded files
- **When created**: Immediately when you upload via `/upload_test` endpoint
- **Naming**: Timestamped filenames (e.g., `20250923_143000_hotel-receipt-25.jpg`)
- **Cleanup**: Files remain until manually deleted (not auto-cleaned)

**2. `data/ocr_output/` - OCR Text Extraction**
- **Purpose**: Stores extracted text from documents using IBM Granite Docling
- **When created**: During OCR processing phase
- **Content**: Plain text files with extracted content from PDFs/images
- **Usage**: Used for document chunking and embedding generation

**3. `data/documents/` - Document Metadata & Chunks**
- **Purpose**: Stores processed document metadata and text chunks
- **When created**: After text extraction and chunking
- **Content**: JSON files with document info, chunks, and processing metadata
- **Usage**: Powers the Claims Processing AI retrieval system

**4. `data/chroma_db/` - Vector Database**
- **Purpose**: ChromaDB vector embeddings for semantic search
- **When created**: During embedding generation phase
- **Content**: Vector embeddings of document chunks for LLM retrieval
- **Usage**: Enables context-aware document querying during extraction tasks

**5. `data/output/` - Processing Results**
- **Purpose**: Final extraction results and summaries
- **When created**: After complete processing of all extraction tasks
- **Content**: JSON and CSV files with extraction results, summaries, and metadata
- **Naming**: Timestamped results (e.g., `processing_results_20250923_143000.json`)

### 🔄 **Processing Flow Example**

```
Upload Document → data/uploads/
    ↓
OCR Processing → data/ocr_output/
    ↓
Text Chunking → data/documents/
    ↓
Vector Embeddings → data/chroma_db/
    ↓
Task Extraction → data/output/
```

### 💡 **Key Points**

- **All folders are created automatically** when you first run the API
- **Documents persist** in uploads folder until manually cleaned
- **Processing is incremental** - each document goes through all stages
- **Memory management** clears temporary data between processing tasks
- **Results are timestamped** for easy tracking and debugging

## Configuration

Edit `config.json` to customize:

```json
{
  "models": {
    "text_model": "phi4-mini:3.8b",
    "embedding_model": "nomic-embed-text"
  },
  "processing": {
    "chunk_size": 300,
    "chunk_overlap": 50,
    "max_context_length": 2500
  },
  "api": {
    "port": 8000,
    "max_upload_size": 10485760
  }
}
```



## Troubleshooting

### Issue: Ollama not running
```bash
# Check if Ollama is installed
ollama --version

# Start Ollama server
ollama serve

# Verify models are available
ollama list
```

### Issue: Out of memory errors
```bash
# Clear memory via API
curl -X POST http://localhost:8000/clear-memory

# Or restart the API server
```

### Issue: Slow processing
- Ensure you're using the recommended models (phi4-mini:3.8b is optimized for speed)
- Check CPU/RAM usage
- Consider processing documents in smaller batches

### Issue: Poor extraction accuracy
- Ensure documents have clear text/images
- For scanned documents, ensure good scan quality
- The system works best with standard business documents (invoices, receipts)

## Performance Optimization

1. **Memory Management**: The system automatically clears memory between documents and tasks

2. **Batch Processing**: Process multiple documents together for efficiency
   ```python
   # Efficient batch processing
   files = [open(f, 'rb') for f in file_list]
   response = requests.post('/process/documents', files=files)
   ```

3. **Chunk Size Tuning**: Adjust chunk size in config.json based on your documents
   - Smaller chunks (200-300): Better for detailed extraction
   - Larger chunks (400-500): Faster processing


   - RAM usage (should stay under 4GB normally)
   - Disk space for ChromaDB
   - API response times

## Project Structure Summary

```
rag_expense_processor/
├── setup.py               # One-time setup script (run with internet)
├── api/                    # FastAPI backend
│   ├── __init__.py
│   └── main.py  # Streamlined API server (2 endpoints)
├── core/                   # Core processing logic
│   ├── __init__.py
│   ├── document_extractor.py # OCR with Docling
│   ├── document_manager.py  # Document chunking & management
│   ├── vector_store.py     # ChromaDB integration
│   └── memory_utils.py     # Memory management
├── config.py              # Configuration management
├── config.json            # System configuration
├── requirements.txt       # Python dependencies
├── README.md              # This documentation
└── data/                  # Runtime data (created automatically)
    ├── uploads/           # Uploaded documents
    ├── output/            # Processing results
    ├── documents/         # Processed document metadata
    ├── ocr_output/        # OCR text outputs
    └── chroma_db/         # Vector database
```

## License

This system uses open-source models and libraries. Ensure compliance with:
- IBM Granite Docling license
- Ollama and model licenses
- ChromaDB and other dependencies

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs` (when server is running)
3. Check logs in the console output
4. Verify all models are properly downloaded

**Remember**: After initial setup, this system works completely offline. No internet connection is required for document processing!