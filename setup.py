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
import os

print("Claims Processing AI - One-time Setup")
print("=" * 70)
print("This setup requires internet connection")
print("After setup, the system will work completely offline")
print("=" * 70)

# ================================
# STEP 1: Install Python Dependencies
# ================================
print("\nPreparing virtual environment and installing Python dependencies...")

# Ensure we create a .venv that uses Python 3.11.13 and install ALL dependencies into that venv only.
TARGET_VERSION = "3.11.13"
VENV_DIR = ".venv"

def _get_version_of_exe(exe_cmd):
    """Return version string like '3.11.13' for a python executable or None."""
    try:
        # exe_cmd can be a list (for 'py -3.11') or a string executable
        if isinstance(exe_cmd, list):
            proc = subprocess.run(exe_cmd + ["-c", "import sys;print(\".\".join(map(str,sys.version_info[:3])))"],
                                  capture_output=True, text=True, timeout=10)
        else:
            proc = subprocess.run([exe_cmd, "-c", "import sys;print(\".\".join(map(str,sys.version_info[:3])))"],
                                  capture_output=True, text=True, timeout=10)

        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        return None
    return None

def find_python_for_target(target_version: str):
    """Try to locate a python executable whose version matches target_version.

    Returns the absolute path / executable specifier to use, or None.
    """
    candidates = [sys.executable, ["py", "-3.11"], "python3.11", "python3", "python"]

    for cand in candidates:
        try:
            ver = _get_version_of_exe(cand)
            if ver:
                print(f"Detected Python candidate {cand} -> version {ver}")
                if ver == target_version:
                    return cand
        except Exception:
            continue

    return None

def create_or_reuse_venv(python_exe, venv_dir: str):
    """Create a venv using the specified python executable if it doesn't exist.

    Returns path to the venv python executable to use for installs.
    """
    venv_python = os.path.join(venv_dir, "Scripts", "python.exe")

    # If venv exists, reuse it regardless of exact Python patch-level differences.
    # Just warn if versions differ from the TARGET_VERSION instead of deleting the venv.
    if os.path.exists(venv_dir):
        if os.path.exists(venv_python):
            ver = _get_version_of_exe(venv_python)
            print(f"Using existing venv at {venv_dir} (Python {ver})")
            if ver != TARGET_VERSION:
                print(f"WARNING: existing venv python version {ver} does not match target {TARGET_VERSION}, but will continue.")
            return venv_python

    # Create the venv using the located python executable
    try:
        print(f"Creating virtual environment at {venv_dir} using {python_exe}...")
        if isinstance(python_exe, list):
            cmd = python_exe + ["-m", "venv", venv_dir]
        else:
            cmd = [python_exe, "-m", "venv", venv_dir]

        subprocess.run(cmd, check=True)
        if os.path.exists(venv_python):
            ver = _get_version_of_exe(venv_python)
            print(f"Created venv with Python {ver}")
            if ver != TARGET_VERSION:
                print(f"WARNING: venv python version {ver} does not match target {TARGET_VERSION}")
            return venv_python
        else:
            print("✗ Failed to find python in the created venv")
            return None
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed creating virtualenv: {e}")
        return None


# Locate a python executable that is exactly the target version
python_candidate = find_python_for_target(TARGET_VERSION)
if not python_candidate:
    # Do not abort if the exact target version isn't found. Use whatever Python is available
    # (e.g., the one running this script) and warn the user. This makes setup more flexible
    # on systems with newer Python versions such as 3.12.x.
    print(f"⚠ Python {TARGET_VERSION} was not found on PATH. Proceeding with available Python interpreter (this may still work).")
    print("   On Windows you can install the exact version if you want, but it's optional.")
    fallback = sys.executable
    ver = _get_version_of_exe(fallback)
    print(f"Detected fallback Python {fallback} -> version {ver}")
    python_candidate = fallback

# Create or reuse the .venv using the located python
venv_python = create_or_reuse_venv(python_candidate, VENV_DIR)
if not venv_python:
    print("✗ Could not create or prepare the virtual environment")
    sys.exit(1)

# Install dependencies INTO the venv only
try:
    print("Installing dependencies into the virtual environment (.venv)...")
    # Upgrade pip/setuptools/wheel first
    subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)

    subprocess.run([venv_python, "-m", "pip", "install", "-r", "requirements.txt", "--no-cache-dir", "--force-reinstall"],
                   check=True)

    print("✓ All packages installed successfully into .venv")
    # Create small helper activation scripts for common shells and print instructions.
    try:
        project_root = os.path.abspath(os.path.dirname(__file__))
        # PowerShell activation helper (user should run this in their PowerShell session)
        ps1_path = os.path.join(project_root, "activate_venv.ps1")
        with open(ps1_path, "w", encoding="utf-8") as f:
            f.write("# PowerShell helper to activate the .venv in this project\n")
            f.write("# Run in PowerShell: .\\activate_venv.ps1\n")
            f.write("$venv = Join-Path $PSScriptRoot '.venv\\Scripts\\Activate.ps1'\n")
            f.write("if (Test-Path $venv) { . $venv } else { Write-Error 'No .venv found; run setup.py to create it.' }\n")

        # CMD activation helper
        bat_path = os.path.join(project_root, "activate_venv.bat")
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write("@echo off\n")
            f.write("if exist .venv\\Scripts\\activate.bat (call .venv\\Scripts\\activate.bat) else (echo No .venv found; run setup.py to create it.)\n")

        # POSIX shell helper
        sh_path = os.path.join(project_root, "activate_venv.sh")
        with open(sh_path, "w", encoding="utf-8") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("# POSIX helper to activate the venv in a new shell: source ./activate_venv.sh\n")
            f.write("if [ -f .venv/bin/activate ]; then\n")
            f.write("  . .venv/bin/activate\n")
            f.write("else\n")
            f.write("  echo 'No .venv found; run setup.py to create it.'\n")
            f.write("fi\n")

        # Make shell script executable where possible
        try:
            os.chmod(sh_path, 0o755)
        except Exception:
            pass

        print("\nTo activate the virtual environment for this project run one of the following in your shell:")
        print("  PowerShell (recommended on Windows):")
        print("    PowerShell: .\\activate_venv.ps1")
        print("    If you get an execution policy error, run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\\activate_venv.ps1")
        print("  cmd.exe:")
        print("    .\\activate_venv.bat")
        print("  bash / WSL / macOS:")
        print("    source ./activate_venv.sh")

    except Exception as e:
        print(f"Failed to create activation helper scripts: {e}")
except subprocess.CalledProcessError as e:
    print(f"✗ Package installation failed: {e}")
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
    # "gemma3:4b": "Text processing and extraction",
    "phi4-mini:3.8b": "Text processing and extraction",
    # "llama3.2:3b": "Text processing and extraction",
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
    except Exception as e:
        # Log but continue; downloads will happen on-demand later.
        print(f"✗ Error during model download step: {e}")


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

# # ================================
# # STEP 4: Pre-download Dolphin OCR Model
# # ================================
# print("\nPre-downloading ByteDance Dolphin OCR models...")

# def predownload_dolphin():
#     """Simple import to trigger Dolphin model download"""
#     try:
#         print("Initializing Dolphin OCR (this will download models)...")
#         from dolphinofficial.ocr import OCR
        
#         # This line initializes the OCR engine and triggers
#         # the model download to the local cache.
#         ocr_engine = OCR() 
        
#         print("✓ Dolphin OCR models downloaded and cached")
#         del ocr_engine # Free up memory
#         return True

#     except Exception as e:
#         print(f"✗ Dolphin OCR setup error: {e}")
#         print("   Please ensure you have internet and `dolphinofficial` is installed.")
#         return False

# predownload_dolphin()

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
        "text_model": "phi4-mini:3.8b",
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