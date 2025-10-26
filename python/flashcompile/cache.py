"""
Compilation cache: Compile once, execute many times
"""

import hashlib
import pickle
import tempfile
from pathlib import Path
from typing import Optional, Callable, Any, Tuple
import numpy as np

class CompilationCache:
    """
    Cache for compiled MLIR modules
    
    Caches compiled code based on:
    - Operation type
    - Input shapes
    - Optimization level
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        if cache_dir is None:
            cache_dir = Path(tempfile.gettempdir()) / "flashcompile_cache"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for this session
        self._memory_cache = {}
    
    def _compute_key(self, operation: str, shapes: Tuple, opt_level: int = 2) -> str:
        """
        Compute cache key for operation
        
        Args:
            operation: Operation name (e.g., "matmul")
            shapes: Input shapes tuple
            opt_level: Optimization level
        
        Returns:
            Cache key string
        """
        # Create deterministic key from operation + shapes + opt_level
        key_data = f"{operation}_{shapes}_{opt_level}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, operation: str, shapes: Tuple, opt_level: int = 2) -> Optional[Callable]:
        """
        Get cached compiled function
        
        Returns:
            Compiled function if cached, None otherwise
        """
        key = self._compute_key(operation, shapes, opt_level)
        
        # Check memory cache first (fastest)
        if key in self._memory_cache:
            return self._memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    compiled_fn = pickle.load(f)
                
                # Store in memory cache
                self._memory_cache[key] = compiled_fn
                return compiled_fn
            except Exception as e:
                # Cache corrupted, remove it
                cache_file.unlink()
                return None
        
        return None
    
    def put(self, operation: str, shapes: Tuple, compiled_fn: Callable, opt_level: int = 2):
        """
        Store compiled function in cache
        
        Args:
            operation: Operation name
            shapes: Input shapes
            compiled_fn: Compiled function
            opt_level: Optimization level
        """
        key = self._compute_key(operation, shapes, opt_level)
        
        # Store in memory cache
        self._memory_cache[key] = compiled_fn
        
        # Store on disk
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(compiled_fn, f)
        except Exception:
            # Caching failed, but continue (memory cache still works)
            pass
    
    def clear(self):
        """Clear all caches"""
        self._memory_cache.clear()
        
        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception:
                pass
    
    def info(self):
        """Print cache statistics"""
        memory_entries = len(self._memory_cache)
        disk_entries = len(list(self.cache_dir.glob("*.pkl")))
        
        print(f"Cache Info:")
        print(f"  Memory entries: {memory_entries}")
        print(f"  Disk entries: {disk_entries}")
        print(f"  Cache directory: {self.cache_dir}")

# Global cache instance
_global_cache = CompilationCache()

def get_cache() -> CompilationCache:
    """Get global compilation cache"""
    return _global_cache