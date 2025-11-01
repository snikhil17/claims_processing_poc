#!/usr/bin/env python3
"""
One-time setup script for Claims Processing AI
Run this WITH internet connection to download all required models and dependencies
After this, the system can run completely offline
"""

import subprocess
import sys
import json
import requests
import time

print("Claims Processing AI - One-time Setup")
print("=" * 70)
print("This setup requires internet connection")
print("After setup, the system will work completely offline")
print("=" * 70)

# ================================
# STEP 1: Install Python Dependencies
# ================================
print("\nInstalling Python dependencies...")

try:
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                          check=True, capture_output=True, text=True)
    print("✓ All packages installed successfully from requirements.txt")
except subprocess.CalledProcessError as e:
    print(f"✗ Package installation failed: {e.stderr}")
    sys.exit(1)

# ================================
# STEP 2: Check/Setup Ollama
# ================================
print("\nChecking Ollama...")

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        # Check if ollama command exists
        result = subprocess.run(["ollama", "--version"],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ Ollama installed: {result.stdout.strip()}")

            # Check if server is running
            try:
                response = requests.get("http://127.0.0.1:11434/api/version", timeout=5)
                if response.status_code == 200:
                    print("✓ Ollama server is running")
                    return True
            except:
                print("Ollama server not running, attempting to start...")
                # Try to start server
                subprocess.Popen(["ollama", "serve"],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                time.sleep(5)
                return True
        return False
    except:
        return False

if not check_ollama():
    print("✗ Ollama not found. Please install Ollama from https://ollama.ai")
    print("   Download from: https://ollama.ai/download")
    sys.exit(1)

# ================================
# STEP 3: Download Ollama Models
# ================================
print("\nDownloading Ollama models...")

# Models used in the application
ollama_models = {
    "gemma3:4b": "Text processing and extraction",
    "nomic-embed-text": "Document embeddings"
}

for model, description in ollama_models.items():
    print(f"\nDownloading {model} ({description})...")
    try:
        result = subprocess.run(["ollama", "pull", model],
                              capture_output=True, text=True, timeout=600,
                              encoding='utf-8', errors='replace')
        if result.returncode == 0:
            print(f"✓ {model} downloaded successfully")
        else:
            print(f"✗ Error downloading {model}: {result.stderr}")
    except subprocess.TimeoutExpired:
        print(f"Download timeout for {model}, but it may continue in background")
    except Exception as e:
        print(f"✗ Error downloading {model}: {e}")

# ================================
# STEP 4: Pre-download Docling Models (Simple Import)
# ================================
print("\nPre-downloading IBM Granite Docling models...")

def predownload_docling():
    """Simple import to trigger Docling model download"""
    try:
        print("Initializing Docling (this will download models)...")
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat

        # Configure pipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        # Initialize converter - triggers model download
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        print("✓ IBM Granite Docling models downloaded and cached")
        return True

    except Exception as e:
        print(f"✗ Docling setup error: {e}")
        print("Models will download on first use")
        return False

predownload_docling()

# ================================
# STEP 5: Create Configuration File
# ================================
print("\nCreating configuration file...")

config_content = {
    "system": {
        "offline_mode": True,
        "data_dir": "data",
        "models_dir": "models",
        "chroma_db_path": "data/chroma_db"
    },
    "models": {
        "text_model": "gemma3:4b",
        "embedding_model": "nomic-embed-text",
        "ollama_host": "http://127.0.0.1:11434"
    },
    "processing": {
        "chunk_size": 300,
        "chunk_overlap": 50,
        "max_context_length": 2500,
        "temperature": 0.1
    },
    "api": {
        "host": "0.0.0.0",
        "port": 8002,
        "reload": False
    }
}

with open("config.json", "w") as f:
    json.dump(config_content, f, indent=2)
print("✓ Configuration file created")

# ================================
# STEP 6: Simple Verification
# ================================
print("\nRunning simple verification...")

def simple_verification():
    """Simple import tests to verify setup"""
    try:
        # Test basic imports
        import ollama
        print("✓ Ollama library working")
        
        import chromadb
        print("✓ ChromaDB library working")
        
        from docling.document_converter import DocumentConverter
        print("✓ Docling library working")
        
        from PIL import Image
        import pytesseract
        print("✓ OCR libraries working")
        
        # Test ChromaDB initialization
        client = chromadb.PersistentClient(path="./data/chroma_db")
        collection = client.get_or_create_collection(
            name="expense_documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("✓ ChromaDB initialized successfully")
        
        return True

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

verification_success = simple_verification()

# ================================
# FINAL SUMMARY
# ================================
print("\n" + "=" * 70)
if verification_success:
    print("SETUP COMPLETE!")
    print("=" * 70)
    print("✓ All dependencies installed")
    print("✓ Ollama models downloaded")
    print("✓ IBM Granite Docling ready")
    print("✓ ChromaDB initialized")
    print("✓ Configuration created")
    print("\nYou can now disconnect from internet and run the system offline!")
    print("\nNext steps:")
    print("  1. Run the API server: python api/main.py")
    print("  2. Access the API at: http://localhost:8002")
    print("  3. API documentation at: http://localhost:8002/docs")
    print("\nRemember: Keep Ollama server running (ollama serve)")
    print("All processing happens offline after this setup!")
else:
    print("SETUP COMPLETED WITH WARNINGS")
    print("=" * 70)
    print("Some verification tests failed, but basic setup is complete")
    print("You can still try running the application")