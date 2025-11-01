"""
Memory management utilities for RAG Expense Processor
"""

import gc
import os
import psutil
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def clear_memory():
    """Clear Python garbage collection and free memory"""
    try:
        # Force garbage collection
        gc.collect()
        logger.info("Memory cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")

def clear_cpu_memory():
    """Clear CPU memory by forcing garbage collection"""
    import gc
    import psutil
    import os
    
    try:
        # Force garbage collection
        gc.collect()
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Get memory info before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Force another garbage collection
        gc.collect()
        
        # Get memory info after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"🧹 CPU Memory cleared: {memory_before:.1f} MB → {memory_after:.1f} MB")
        
    except Exception as e:
        print(f"⚠️ Error clearing CPU memory: {e}")

def clear_gpu_memory():
    """Clear GPU memory if available"""
    try:
        import torch
        if torch.cuda.is_available():
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Get GPU memory info
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            gpu_cached = torch.cuda.memory_reserved(0) / 1024**3  # GB
            
            print(f"🧹 GPU Memory cleared - Total: {gpu_memory:.1f}GB, Allocated: {gpu_allocated:.1f}GB, Cached: {gpu_cached:.1f}GB")
        else:
            print("⚠️ No GPU available for memory clearing")
    except ImportError:
        print("⚠️ PyTorch not available - GPU memory clearing skipped")
    except Exception as e:
        print(f"⚠️ Error clearing GPU memory: {e}")

def force_memory_cleanup():
    """Force comprehensive memory cleanup"""
    print("🧹 Starting memory cleanup...")
    clear_cpu_memory()
    clear_gpu_memory()
    print("✅ Memory cleanup complete")

def get_memory_stats() -> Dict[str, Any]:
    """Get current memory usage statistics"""
    try:
        # Get process memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Get system memory info
        virtual_memory = psutil.virtual_memory()
        
        stats = {
            "process_memory_mb": round(memory_info.rss / 1024 / 1024, 2),
            "process_memory_percent": round(process.memory_percent(), 2),
            "system_memory_total_mb": round(virtual_memory.total / 1024 / 1024, 2),
            "system_memory_available_mb": round(virtual_memory.available / 1024 / 1024, 2),
            "system_memory_percent": virtual_memory.percent,
            "cpu_percent": psutil.cpu_percent(interval=1)
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        return {"error": str(e)}
