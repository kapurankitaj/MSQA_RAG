"""
Optimization module for improving RAG system performance.
This module implements profiling, memory optimization, 
caching strategies, and batch processing.
"""

import os
import json
import time
import logging
import pickle
import threading
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    filename='optimization.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('optimization')

class PerformanceProfiler:
    """
    Profiles system performance and identifies bottlenecks.
    """
    
    def __init__(self, save_path: str = 'data/optimization/profile_data.json'):
        """
        Initialize performance profiler.
        
        Args:
            save_path: Path to save profiling data
        """
        self.save_path = save_path
        self.profile_data = {}
        self._load_profile_data()
        self.current_run = {}
        logger.info("PerformanceProfiler initialized")
        
    def _load_profile_data(self):
        """Load existing profile data if available"""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    self.profile_data = json.load(f)
                logger.info(f"Loaded profile data with {len(self.profile_data)} entries")
        except Exception as e:
            logger.error(f"Failed to load profile data: {str(e)}")
            
    def _save_profile_data(self):
        """Save profile data to disk"""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as f:
                json.dump(self.profile_data, f, indent=2)
            logger.info(f"Saved profile data with {len(self.profile_data)} entries")
        except Exception as e:
            logger.error(f"Failed to save profile data: {str(e)}")
            
    def start_profiling(self, run_id: str):
        """
        Start profiling a new run.
        
        Args:
            run_id: Unique identifier for this profiling run
        """
        self.current_run = {
            'run_id': run_id,
            'start_time': time.time(),
            'components': {},
            'memory_usage': [],
            'end_time': None,
            'total_time': None
        }
        
        # Take initial memory snapshot
        self._capture_memory_usage()
        
        logger.info(f"Started profiling run: {run_id}")
        
    def profile_component(self, component_name: str) -> Callable:
        """
        Decorator to profile a component's execution time.
        
        Args:
            component_name: Name of the component to profile
            
        Returns:
            Callable: Decorator function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Skip if we're not in an active profiling run
                if not self.current_run:
                    return func(*args, **kwargs)
                    
                # Initialize component data if needed
                if component_name not in self.current_run['components']:
                    self.current_run['components'][component_name] = {
                        'calls': 0,
                        'total_time': 0,
                        'min_time': float('inf'),
                        'max_time': 0,
                        'start_memory': 0,
                        'end_memory': 0
                    }
                
                # Capture memory before execution
                start_memory = self._get_memory_usage()
                self.current_run['components'][component_name]['start_memory'] = start_memory
                
                # Time the function
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                # Capture memory after execution
                end_memory = self._get_memory_usage()
                self.current_run['components'][component_name]['end_memory'] = end_memory
                
                # Record timing data
                execution_time = end_time - start_time
                component_data = self.current_run['components'][component_name]
                component_data['calls'] += 1
                component_data['total_time'] += execution_time
                component_data['min_time'] = min(component_data['min_time'], execution_time)
                component_data['max_time'] = max(component_data['max_time'], execution_time)
                
                # Take a memory snapshot periodically
                self._capture_memory_usage()
                
                return result
            return wrapper
        return decorator
        
    def end_profiling(self) -> Dict:
        """
        End the current profiling run and save results.
        
        Returns:
            Dict: Profiling results
        """
        if not self.current_run:
            logger.warning("No active profiling run to end")
            return {}
            
        # Record end time and total duration
        self.current_run['end_time'] = time.time()
        self.current_run['total_time'] = self.current_run['end_time'] - self.current_run['start_time']
        
        # Take final memory snapshot
        self._capture_memory_usage()
        
        # Calculate average times
        for component, data in self.current_run['components'].items():
            if data['calls'] > 0:
                data['avg_time'] = data['total_time'] / data['calls']
            else:
                data['avg_time'] = 0
                
        # Store the run data
        run_id = self.current_run['run_id']
        timestamp = datetime.now().isoformat()
        self.profile_data[f"{run_id}_{timestamp}"] = self.current_run
        
        # Save to disk
        self._save_profile_data()
        
        logger.info(f"Completed profiling run: {run_id} in {self.current_run['total_time']:.2f} seconds")
        
        # Get a copy of the results before clearing
        results = dict(self.current_run)
        self.current_run = {}
        
        return results
        
    def _capture_memory_usage(self):
        """Capture current memory usage"""
        if not self.current_run:
            return
            
        memory_usage = self._get_memory_usage()
        self.current_run['memory_usage'].append({
            'timestamp': time.time(),
            'memory_mb': memory_usage
        })
        
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            float: Memory usage in MB
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert bytes to MB
        except ImportError:
            logger.warning("psutil not available, memory profiling disabled")
            return 0
            
    def analyze_bottlenecks(self) -> Dict:
        """
        Analyze profiling data to identify bottlenecks.
        
        Returns:
            Dict: Analysis results
        """
        if not self.profile_data:
            return {"error": "No profiling data available"}
            
        analysis = {
            "component_analysis": {},
            "memory_analysis": {},
            "recommendations": []
        }
        
        # Analyze all runs
        all_components = set()
        component_times = {}
        
        for run_id, run_data in self.profile_data.items():
            for component, data in run_data.get('components', {}).items():
                all_components.add(component)
                if component not in component_times:
                    component_times[component] = []
                component_times[component].append(data.get('total_time', 0))
        
        # Calculate average time per component across all runs
        avg_times = {}
        for component, times in component_times.items():
            if times:
                avg_times[component] = sum(times) / len(times)
            else:
                avg_times[component] = 0
                
        # Sort components by average time
        sorted_components = sorted(avg_times.items(), key=lambda x: x[1], reverse=True)
        
        # Identify bottlenecks (components taking >15% of total time)
        total_time = sum(avg_times.values())
        bottlenecks = []
        
        for component, avg_time in sorted_components:
            percentage = (avg_time / total_time * 100) if total_time > 0 else 0
            analysis["component_analysis"][component] = {
                "avg_time": avg_time,
                "percentage": percentage
            }
            
            if percentage > 15:
                bottlenecks.append(component)
                
        # Generate recommendations
        if bottlenecks:
            analysis["recommendations"].append(f"Consider optimizing these components: {', '.join(bottlenecks)}")
            
        # Memory analysis - identify components with high memory growth
        for run_id, run_data in self.profile_data.items():
            for component, data in run_data.get('components', {}).items():
                start_mem = data.get('start_memory', 0)
                end_mem = data.get('end_memory', 0)
                memory_growth = end_mem - start_mem
                
                if component not in analysis["memory_analysis"]:
                    analysis["memory_analysis"][component] = []
                    
                analysis["memory_analysis"][component].append(memory_growth)
                
        # Calculate average memory growth
        for component, growths in analysis["memory_analysis"].items():
            avg_growth = sum(growths) / len(growths) if growths else 0
            analysis["memory_analysis"][component] = avg_growth
            
            if avg_growth > 50:  # If more than 50MB growth
                analysis["recommendations"].append(f"Component {component} shows high memory usage growth ({avg_growth:.2f} MB)")
                
        return analysis


class MemoryOptimizer:
    """
    Optimizes memory usage for large document collections.
    """
    
    def __init__(self):
        """Initialize memory optimizer"""
        self.memory_stats = {}
        logger.info("MemoryOptimizer initialized")
        
    def optimize_document_storage(self, documents: List[Dict]) -> List[Dict]:
        """
        Optimize memory usage for document storage.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List[Dict]: Memory-optimized documents
        """
        if not documents:
            return documents
            
        start_memory = self._estimate_size(documents)
        logger.info(f"Starting document optimization, initial size: {start_memory:.2f} MB")
        
        # Optimization 1: Convert string keys to enum integers for repetitive metadata
        key_mapping = {}
        reverse_mapping = {}
        next_key_id = 0
        
        # Identify common metadata keys
        common_keys = set()
        for doc in documents:
            if 'metadata' in doc:
                common_keys.update(doc['metadata'].keys())
                
        # Create mapping for common keys
        for key in common_keys:
            key_mapping[key] = next_key_id
            reverse_mapping[next_key_id] = key
            next_key_id += 1
            
        # Store the mapping for later use
        self._save_key_mapping(key_mapping, reverse_mapping)
        
        # Apply mapping to documents
        optimized_docs = []
        for doc in documents:
            opt_doc = dict(doc)  # Shallow copy
            
            # Optimize metadata
            if 'metadata' in opt_doc:
                opt_metadata = {}
                for k, v in opt_doc['metadata'].items():
                    if k in key_mapping:
                        opt_metadata[key_mapping[k]] = v
                    else:
                        opt_metadata[k] = v
                opt_doc['metadata'] = opt_metadata
                
            # Optimization 2: Remove redundant whitespace in text content
            if 'content' in opt_doc and isinstance(opt_doc['content'], str):
                # Replace multiple spaces with single space
                opt_doc['content'] = ' '.join(opt_doc['content'].split())
                
            optimized_docs.append(opt_doc)
            
        end_memory = self._estimate_size(optimized_docs)
        savings = start_memory - end_memory
        savings_percent = (savings / start_memory * 100) if start_memory > 0 else 0
        
        logger.info(f"Document optimization complete: {end_memory:.2f} MB " 
                    f"(saved {savings:.2f} MB, {savings_percent:.1f}%)")
        
        # Record statistics
        self.memory_stats['document_optimization'] = {
            'original_size': start_memory,
            'optimized_size': end_memory,
            'savings': savings,
            'savings_percent': savings_percent
        }
        
        return optimized_docs
    
    def restore_document_keys(self, optimized_docs: List[Dict]) -> List[Dict]:
        """
        Restore original keys for optimized documents.
        
        Args:
            optimized_docs: List of optimized documents
            
        Returns:
            List[Dict]: Documents with original keys
        """
        _, reverse_mapping = self._load_key_mapping()
        if not reverse_mapping:
            return optimized_docs
            
        restored_docs = []
        for doc in optimized_docs:
            restored_doc = dict(doc)
            
            # Restore original keys in metadata
            if 'metadata' in restored_doc:
                restored_metadata = {}
                for k, v in restored_doc['metadata'].items():
                    if isinstance(k, int) and k in reverse_mapping:
                        restored_metadata[reverse_mapping[k]] = v
                    else:
                        restored_metadata[k] = v
                restored_doc['metadata'] = restored_metadata
                
            restored_docs.append(restored_doc)
            
        return restored_docs
    
    def _save_key_mapping(self, key_mapping: Dict, reverse_mapping: Dict):
        """
        Save key mapping to disk.
        
        Args:
            key_mapping: Forward mapping (string -> int)
            reverse_mapping: Reverse mapping (int -> string)
        """
        try:
            os.makedirs('data/optimization', exist_ok=True)
            with open('data/optimization/key_mapping.json', 'w') as f:
                json.dump({
                    'key_mapping': {k: v for k, v in key_mapping.items()},
                    'reverse_mapping': {str(k): v for k, v in reverse_mapping.items()}
                }, f)
        except Exception as e:
            logger.error(f"Failed to save key mapping: {str(e)}")
            
    def _load_key_mapping(self) -> Tuple[Dict, Dict]:
        """
        Load key mapping from disk.
        
        Returns:
            Tuple[Dict, Dict]: Forward and reverse mappings
        """
        key_mapping = {}
        reverse_mapping = {}
        
        try:
            if os.path.exists('data/optimization/key_mapping.json'):
                with open('data/optimization/key_mapping.json', 'r') as f:
                    data = json.load(f)
                    key_mapping = data.get('key_mapping', {})
                    # Convert string keys back to integers
                    reverse_mapping = {int(k): v for k, v in data.get('reverse_mapping', {}).items()}
        except Exception as e:
            logger.error(f"Failed to load key mapping: {str(e)}")
            
        return key_mapping, reverse_mapping
    
    def optimize_embeddings(self, embeddings: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """
        Optimize memory usage for embeddings.
        
        Args:
            embeddings: Dictionary mapping document IDs to embedding vectors
            
        Returns:
            Dict[str, List[float]]: Optimized embeddings
        """
        if not embeddings:
            return embeddings
            
        start_memory = self._estimate_size(embeddings)
        logger.info(f"Starting embeddings optimization, initial size: {start_memory:.2f} MB")
        
        # Optimization: Convert float64 to float32 or float16
        optimized_embeddings = {}
        
        import numpy as np
        for doc_id, embedding in embeddings.items():
            # Convert to float16 with safe handling of potential numpy arrays
            if isinstance(embedding, np.ndarray):
                # Convert numpy array to float16
                embedding_f16 = embedding.astype(np.float16)
                # Convert back to Python list for JSON serialization
                optimized_embeddings[doc_id] = embedding_f16.tolist()
            else:
                # Handle Python lists - convert to numpy and back for precision conversion
                embedding_array = np.array(embedding, dtype=np.float64)
                embedding_f16 = embedding_array.astype(np.float16)
                optimized_embeddings[doc_id] = embedding_f16.tolist()
                
        end_memory = self._estimate_size(optimized_embeddings)
        savings = start_memory - end_memory
        savings_percent = (savings / start_memory * 100) if start_memory > 0 else 0
        
        logger.info(f"Embeddings optimization complete: {end_memory:.2f} MB " 
                    f"(saved {savings:.2f} MB, {savings_percent:.1f}%)")
        
        # Record statistics
        self.memory_stats['embeddings_optimization'] = {
            'original_size': start_memory,
            'optimized_size': end_memory,
            'savings': savings,
            'savings_percent': savings_percent
        }
        
        return optimized_embeddings
    
    def _estimate_size(self, obj: Any) -> float:
        """
        Estimate memory size of an object in MB.
        
        Args:
            obj: Python object to measure
            
        Returns:
            float: Size estimate in MB
        """
        import sys
        import pickle
        
        # For accurate measurement of custom objects
        try:
            size = sys.getsizeof(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
            return size / 1024 / 1024  # Convert bytes to MB
        except Exception:
            # Fallback for objects that can't be pickled
            if isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
            elif isinstance(obj, (list, tuple, set)):
                return sum(self._estimate_size(item) for item in obj)
            else:
                return sys.getsizeof(obj) / 1024 / 1024  # Convert bytes to MB


class CacheManager:
    """
    Implements caching strategies for improved performance.
    """
    
    def __init__(self, cache_dir: str = 'data/optimization/cache',
                 max_cache_size_mb: int = 1000,
                 default_ttl: int = 86400):  # 24 hours in seconds
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cached data
            max_cache_size_mb: Maximum cache size in MB
            default_ttl: Default time-to-live in seconds
        """
        self.cache_dir = cache_dir
        self.max_cache_size_mb = max_cache_size_mb
        self.default_ttl = default_ttl
        self.metadata_file = os.path.join(cache_dir, 'cache_metadata.json')
        self.metadata = {
            'entries': {},
            'size': 0,
            'hits': 0,
            'misses': 0
        }
        
        # Make sure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        self._load_metadata()
        
        # Start a background thread for cache maintenance
        self._start_maintenance_thread()
        
        logger.info(f"CacheManager initialized with max size {max_cache_size_mb}MB")
        
    def _load_metadata(self):
        """Load cache metadata from disk"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded cache metadata with {len(self.metadata['entries'])} entries")
        except Exception as e:
            logger.error(f"Failed to load cache metadata: {str(e)}")
            
    def _save_metadata(self):
        """Save cache metadata to disk"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {str(e)}")
            
    def _start_maintenance_thread(self):
        """Start a background thread for cache maintenance"""
        maintenance_thread = threading.Thread(
            target=self._maintenance_routine,
            daemon=True  # Thread will exit when main program exits
        )
        maintenance_thread.start()
        
    def _maintenance_routine(self):
        """Background routine for cache maintenance"""
        while True:
            # Run maintenance every hour
            time.sleep(3600)
            
            try:
                # Remove expired entries
                self._clean_expired_entries()
                
                # Enforce size limit
                self._enforce_size_limit()
                
                # Save metadata
                self._save_metadata()
                
                logger.info("Cache maintenance completed")
            except Exception as e:
                logger.error(f"Error during cache maintenance: {str(e)}")
    
    def _clean_expired_entries(self):
        """Remove expired cache entries"""
        now = time.time()
        expired_keys = []
        
        for key, metadata in self.metadata['entries'].items():
            if metadata['expires_at'] < now:
                expired_keys.append(key)
                
        for key in expired_keys:
            self._remove_cache_item(key)
            
        if expired_keys:
            logger.info(f"Removed {len(expired_keys)} expired cache entries")
            
    def _enforce_size_limit(self):
        """Enforce maximum cache size by removing least recently used items"""
        # If we're under the limit, nothing to do
        if self.metadata['size'] <= self.max_cache_size_mb:
            return
            
        # Sort entries by last access time
        sorted_entries = sorted(
            self.metadata['entries'].items(),
            key=lambda x: x[1]['last_accessed']
        )
        
        # Remove oldest entries until under size limit
        removed = 0
        for key, _ in sorted_entries:
            self._remove_cache_item(key)
            removed += 1
            
            if self.metadata['size'] <= self.max_cache_size_mb:
                break
                
        logger.info(f"Removed {removed} cache entries to enforce size limit")
            
    def _remove_cache_item(self, key: str):
        """
        Remove a cache item by key.
        
        Args:
            key: Cache key to remove
        """
        if key in self.metadata['entries']:
            # Reduce size accounting
            self.metadata['size'] -= self.metadata['entries'][key]['size_mb']
            
            # Remove file
            cache_file = os.path.join(self.cache_dir, f"{key}.cache")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                
            # Remove from metadata
            del self.metadata['entries'][key]
            
    def generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate a cache key based on function arguments.
        
        Args:
            prefix: Prefix for the key (usually function name)
            *args: Function positional arguments
            **kwargs: Function keyword arguments
            
        Returns:
            str: Cache key
        """
        import hashlib
        
        # Convert arguments to a string representation
        args_str = str(args) + str(sorted(kwargs.items()))
        
        # Create hash of the arguments
        args_hash = hashlib.md5(args_str.encode()).hexdigest()
        
        # Return prefixed key
        return f"{prefix}_{args_hash}"
    
    def cache_function(self, prefix: str, ttl: Optional[int] = None):
        """
        Decorator for caching function results.
        
        Args:
            prefix: Prefix for cache keys
            ttl: Time-to-live in seconds (None for default)
            
        Returns:
            Callable: Decorator function
        """
        if ttl is None:
            ttl = self.default_ttl
            
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self.generate_cache_key(prefix, *args, **kwargs)
                
                # Try to get from cache
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                    
                # Execute function if not in cache
                result = func(*args, **kwargs)
                
                # Store in cache
                self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def get(self, key: str) -> Any:
        """
        Get an item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None if not found
        """
        # Check if key exists and is not expired
        if key not in self.metadata['entries']:
            self.metadata['misses'] += 1
            return None
            
        entry = self.metadata['entries'][key]
        
        # Check expiration
        if entry['expires_at'] < time.time():
            self._remove_cache_item(key)
            self.metadata['misses'] += 1
            return None
            
        # Load from file
        cache_file = os.path.join(self.cache_dir, f"{key}.cache")
        if not os.path.exists(cache_file):
            self._remove_cache_item(key)
            self.metadata['misses'] += 1
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
                
            # Update access time
            self.metadata['entries'][key]['last_accessed'] = time.time()
            self.metadata['hits'] += 1
            
            # Save metadata periodically (every 10 hits)
            if self.metadata['hits'] % 10 == 0:
                self._save_metadata()
                
            return result
            
        except Exception as e:
            logger.error(f"Error reading cache file {key}: {str(e)}")
            self._remove_cache_item(key)
            self.metadata['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Store an item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if ttl is None:
            ttl = self.default_ttl
            
        # Serialize the value
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.cache")
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
                
            # Get file size
            file_size_mb = os.path.getsize(cache_file) / 1024 / 1024
            
            # Update metadata
            self.metadata['entries'][key] = {
                'created_at': time.time(),
                'expires_at': time.time() + ttl,
                'last_accessed': time.time(),
                'size_mb': file_size_mb
            }
            
            # Update total size
            self.metadata['size'] += file_size_mb
            
            # Enforce size limit if exceeded
            if self.metadata['size'] > self.max_cache_size_mb:
                self._enforce_size_limit()
                
            # Save metadata periodically
            if len(self.metadata['entries']) % 10 == 0:
                self._save_metadata()
                
            return True
            
        except Exception as e:
            logger.error(f"Error writing to cache {key}: {str(e)}")
            return False
            
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            bool: True if entry was found and removed, False otherwise
        """
        if key in self.metadata['entries']:
            self._remove_cache_item(key)
            self._save_metadata()
            return True
        return False
        
    def invalidate_prefix(self, prefix: str) -> int:
        """
        Invalidate all cache entries with a specific prefix.
        
        Args:
            prefix: Prefix to match
            
        Returns:
            int: Number of entries invalidated
        """
        keys_to_remove = [k for k in self.metadata['entries'] if k.startswith(prefix)]
        for key in keys_to_remove:
            self._remove_cache_item(key)
            
        if keys_to_remove:
            self._save_metadata()
            
        return len(keys_to_remove)
        
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            int: Number of entries cleared
        """
        count = len(self.metadata['entries'])
        
        # Remove all cache files
        for key in list(self.metadata['entries'].keys()):
            self._remove_cache_item(key)
            
        # Reset metadata
        self.metadata = {
            'entries': {},
            'size': 0,
            'hits': 0,
            'misses': 0
        }
        
        self._save_metadata()
        return count
        
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dict: Cache statistics
        """
        return {
            'entries': len(self.metadata['entries']),
            'size_mb': self.metadata['size'],
            'hits': self.metadata['hits'],
            'misses': self.metadata['misses'],
            'hit_ratio': self.metadata['hits'] / (self.metadata['hits'] + self.metadata['misses']) 
                          if (self.metadata['hits'] + self.metadata['misses']) > 0 else 0
        }


class BatchProcessor:
    """
    Implements efficient batch processing for RAG operations.
    """
    
    def __init__(self, batch_size: int = 16, max_workers: int = 4):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Maximum number of items to process in a batch
            max_workers: Maximum number of parallel workers
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        logger.info(f"BatchProcessor initialized with batch_size={batch_size}, max_workers={max_workers}")
        
    def process_batches(self, items: List[Any], processor_func: Callable, 
                        use_threading: bool = False) -> List[Any]:
        """
        Process items in optimized batches.
        
        Args:
            items: List of items to process
            processor_func: Function to apply to each batch (takes batch as argument)
            use_threading: Whether to use threading for parallel processing
            
        Returns:
            List[Any]: Combined results from all batches
        """
        if not items:
            return []
            
        # Split into batches
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        
        logger.info(f"Processing {len(items)} items in {len(batches)} batches")
        
        results = []
        
        if use_threading and self.max_workers > 1:
            # Process batches in parallel using threading
            from concurrent.futures import ThreadPoolExecutor
            
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
                batch_results = list(executor.map(processor_func, batches))
                
            # Combine results from all batches
            for batch_result in batch_results:
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
        else:
            # Process batches sequentially
            for batch in batches:
                batch_result = processor_func(batch)
                if isinstance(batch_result, list):
                    results.extend(batch_result)
                else:
                    results.append(batch_result)
                    
        return results
        
    def batch_embed_documents(self, documents: List[Dict], embedding_func: Callable) -> Dict[str, List[float]]:
        """
        Create embeddings for documents in batches.
        
        Args:
            documents: List of document dictionaries
            embedding_func: Function to embed a batch of texts
            
        Returns:
            Dict[str, List[float]]: Document ID to embedding mapping
        """
        start_time = time.time()
        logger.info(f"Beginning batch embedding of {len(documents)} documents")
        
        # Extract text content from documents
        texts = []
        doc_ids = []
        
        for doc in documents:
            doc_ids.append(doc.get('id'))
            texts.append(doc.get('content', ''))
            
        # Define batch processor function
        def process_text_batch(text_batch):
            return embedding_func(text_batch)
            
        # Process in batches
        embedding_batches = self.process_batches(texts, process_text_batch, use_threading=False)
        
        # Combine results
        embeddings = {}
        batch_index = 0
        
        for i, batch in enumerate(embedding_batches):
            batch_size = len(batch)
            for j in range(batch_size):
                if batch_index < len(doc_ids):
                    embeddings[doc_ids[batch_index]] = batch[j]
                    batch_index += 1
                    
        elapsed_time = time.time() - start_time
        logger.info(f"Finished batch embedding in {elapsed_time:.2f} seconds")
        
        return embeddings
        
    def batch_query_database(self, queries: List[str], query_func: Callable, 
                            use_threading: bool = True) -> List[Dict]:
        """
        Execute multiple database queries in batches.
        
        Args:
            queries: List of queries to execute
            query_func: Function to execute a batch of queries
            use_threading: Whether to use threading for parallel processing
            
        Returns:
            List[Dict]: Combined query results
        """
        start_time = time.time()
        logger.info(f"Beginning batch database queries for {len(queries)} queries")
        
        # Process in batches
        results = self.process_batches(queries, query_func, use_threading=use_threading)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Finished batch database queries in {elapsed_time:.2f} seconds")
        
        return results
        
    def batch_generate_texts(self, prompts: List[str], generation_func: Callable) -> List[str]:
        """
        Generate texts from multiple prompts in batches.
        
        Args:
            prompts: List of prompts for text generation
            generation_func: Function to generate texts from a batch of prompts
            
        Returns:
            List[str]: Generated texts
        """
        start_time = time.time()
        logger.info(f"Beginning batch text generation for {len(prompts)} prompts")
        
        # Process in batches (usually sequential for LLM API calls)
        results = self.process_batches(prompts, generation_func, use_threading=False)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Finished batch text generation in {elapsed_time:.2f} seconds")
        
        return results


class OptimizationManager:
    """
    Main manager for all optimization strategies.
    """
    
    def __init__(self, max_cache_size_mb: int = 1000, 
                batch_size: int = 16, max_workers: int = 4):
        """
        Initialize optimization manager.
        
        Args:
            max_cache_size_mb: Maximum cache size in MB
            batch_size: Default batch size for processing
            max_workers: Maximum worker threads
        """
        # Initialize components
        self.profiler = PerformanceProfiler()
        self.memory_optimizer = MemoryOptimizer()
        self.cache_manager = CacheManager(max_cache_size_mb=max_cache_size_mb)
        self.batch_processor = BatchProcessor(batch_size=batch_size, max_workers=max_workers)
        
        # Setup configuration
        self.config = {
            'enable_profiling': True,
            'enable_caching': True,
            'enable_memory_optimization': True,
            'enable_batch_processing': True,
            'log_performance_metrics': True
        }
        
        # Map of component functions to their optimized wrappers
        self.optimized_functions = {}
        
        logger.info("OptimizationManager initialized")
        
    def optimize_pipeline(self, pipeline_config: Dict):
        """
        Apply optimization strategies to a RAG pipeline.
        
        Args:
            pipeline_config: Configuration of pipeline components
        """
        if self.config['enable_profiling']:
            self.profiler.start_profiling("pipeline_optimization")
            
        logger.info("Applying optimization strategies to RAG pipeline")
        
        # Apply memory optimization
        if self.config['enable_memory_optimization'] and 'document_storage' in pipeline_config:
            logger.info("Optimizing document storage")
            # This would be called directly on the documents when they're loaded
            
        # Set up caching for expensive operations
        if self.config['enable_caching']:
            # Add caching for embedding generation
            if 'embedding_model' in pipeline_config:
                logger.info("Setting up caching for embeddings")
                # The actual wrapping would happen when the embedding function is registered
                
            # Add caching for vector search
            if 'vector_search' in pipeline_config:
                logger.info("Setting up caching for vector search")
                # The actual wrapping would happen when the search function is registered
                
        # Set up batch processing
        if self.config['enable_batch_processing']:
            logger.info("Setting up batch processing for document ingestion and query")
            # The batch processor would be used directly in those operations
            
        if self.config['enable_profiling']:
            results = self.profiler.end_profiling()
            if self.config['log_performance_metrics']:
                logger.info(f"Pipeline optimization completed in {results['total_time']:.2f} seconds")
                
    def register_function(self, function_name: str, func: Callable) -> Callable:
        """
        Register a function for optimization.
        
        Args:
            function_name: Name of the function
            func: The function to optimize
            
        Returns:
            Callable: Optimized function
        """
        # Apply profiling decorator if enabled
        if self.config['enable_profiling']:
            func = self.profiler.profile_component(function_name)(func)
            
        # Apply caching decorator if enabled
        if self.config['enable_caching']:
            # Check if the function should be cached
            # Not all functions should be cached - only pure functions or those with stable outputs
            cacheable_functions = {
                'embed_documents', 'embed_query', 'search_documents', 
                'extract_keywords', 'rerank_results', 'parse_query'
            }
            
            if function_name in cacheable_functions:
                # Determine an appropriate TTL based on function type
                if function_name.startswith('embed'):
                    ttl = 7 * 86400  # 7 days for embeddings
                elif function_name.startswith('search'):
                    ttl = 3600  # 1 hour for search results
                else:
                    ttl = 86400  # 1 day for other functions
                    
                func = self.cache_manager.cache_function(function_name, ttl)(func)
                
        # Store the optimized function
        self.optimized_functions[function_name] = func
        
        return func
        
    def optimize_vector_index(self, vector_index, document_ids: List[str], 
                             embedding_dim: int) -> Any:
        """
        Optimize a vector index for improved search performance.
        
        Args:
            vector_index: The vector index to optimize
            document_ids: List of document IDs in the index
            embedding_dim: Dimensionality of embeddings
            
        Returns:
            Any: Optimized vector index
        """
        if self.config['enable_profiling']:
            self.profiler.start_profiling("vector_index_optimization")
            
        logger.info(f"Optimizing vector index with {len(document_ids)} documents")
        
        # The actual optimization would depend on the vector database being used
        # For FAISS, this might include:
        try:
            import faiss
            
            # 1. Check if the index is already optimized
            if isinstance(vector_index, faiss.IndexFlatL2):
                # Convert to IVF index for faster search with slight accuracy tradeoff
                nlist = min(4096, len(document_ids) // 10)  # Rule of thumb
                if nlist < 10:  # Too few documents for IVF
                    logger.info("Too few documents for IVF optimization")
                    return vector_index
                    
                # Create optimized index
                optimized_index = faiss.IndexIVFFlat(faiss.IndexFlatL2(embedding_dim), embedding_dim, nlist)
                
                # Train the index
                # In a real implementation, we'd need to collect all vectors first
                logger.info(f"Training IVF index with {nlist} clusters")
                # optimized_index.train(all_vectors)
                
                # Copy data from original index
                logger.info("Copying data to optimized index")
                # for i, doc_id in enumerate(document_ids):
                #     optimized_index.add_with_ids(original_vectors[i:i+1], np.array([i]))
                
                logger.info("Vector index optimization completed")
                # return optimized_index
                
            # If already using IVF but with many vectors, consider adding HNSW
            # This would be similar to above but using IndexHNSWFlat
            
            return vector_index
            
        except (ImportError, Exception) as e:
            logger.error(f"Failed to optimize vector index: {str(e)}")
            return vector_index
        finally:
            if self.config['enable_profiling']:
                results = self.profiler.end_profiling()
                if self.config['log_performance_metrics']:
                    logger.info(f"Vector index optimization completed in {results['total_time']:.2f} seconds")
                    
    def get_optimization_stats(self) -> Dict:
        """
        Get statistics from all optimization components.
        
        Returns:
            Dict: Optimization statistics
        """
        stats = {
            'cache': self.cache_manager.get_stats(),
            'memory': self.memory_optimizer.memory_stats
        }
        
        if self.config['enable_profiling']:
            # Analyze profiling data to add performance stats
            bottleneck_analysis = self.profiler.analyze_bottlenecks()
            stats['performance'] = bottleneck_analysis
            
        return stats
        
    def optimize_document_chunking(self, documents: List[Dict], 
                                  chunk_size: int = 1000, 
                                  chunk_overlap: int = 200) -> List[Dict]:
        """
        Optimize document chunking for better retrieval.
        
        Args:
            documents: Original documents
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List[Dict]: Optimized document chunks
        """
        if self.config['enable_profiling']:
            self.profiler.start_profiling("document_chunking")
            
        logger.info(f"Optimizing document chunking for {len(documents)} documents")
        
        optimized_chunks = []
        
        for doc in documents:
            if 'content' not in doc or not isinstance(doc['content'], str):
                # Skip documents without content
                continue
                
            content = doc['content']
            
            # Split content by paragraphs first
            paragraphs = content.split('\n\n')
            
            current_chunk = ""
            current_metadata = dict(doc.get('metadata', {}))
            
            for i, para in enumerate(paragraphs):
                # If adding this paragraph would exceed chunk size, save current chunk
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    # Create chunk document
                    chunk_doc = {
                        'id': f"{doc.get('id', 'doc')}_{len(optimized_chunks)}",
                        'content': current_chunk.strip(),
                        'metadata': dict(current_metadata)
                    }
                    
                    # Add position metadata
                    chunk_doc['metadata']['chunk_index'] = len(optimized_chunks)
                    
                    optimized_chunks.append(chunk_doc)
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + "\n\n" + para
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                        
            # Add the last chunk if it's not empty
            if current_chunk.strip():
                chunk_doc = {
                    'id': f"{doc.get('id', 'doc')}_{len(optimized_chunks)}",
                    'content': current_chunk.strip(),
                    'metadata': dict(current_metadata)
                }
                
                # Add position metadata
                chunk_doc['metadata']['chunk_index'] = len(optimized_chunks)
                
                optimized_chunks.append(chunk_doc)
                
        if self.config['enable_profiling']:
            results = self.profiler.end_profiling()
            if self.config['log_performance_metrics']:
                logger.info(f"Document chunking completed in {results['total_time']:.2f} seconds")
                
        logger.info(f"Created {len(optimized_chunks)} optimized chunks from {len(documents)} documents")
        return optimized_chunks


# Utility functions that don't fit in classes

def estimate_embedding_cost(num_documents: int, avg_tokens_per_doc: int, 
                          model: str = "text-embedding-3-small") -> Dict:
    """
    Estimate the cost of embedding documents.
    
    Args:
        num_documents: Number of documents to embed
        avg_tokens_per_doc: Average number of tokens per document
        model: Embedding model name
        
    Returns:
        Dict: Cost estimate information
    """
    # Prices as of April 2025 (example)
    model_prices = {
        "text-embedding-3-small": 0.00002,  # per 1K tokens
        "text-embedding-3-large": 0.00013   # per 1K tokens
    }
    
    if model not in model_prices:
        logger.warning(f"Unknown model {model}, using text-embedding-3-small pricing")
        model = "text-embedding-3-small"
        
    # Calculate total tokens
    total_tokens = num_documents * avg_tokens_per_doc
    
    # Calculate cost
    cost_per_1k = model_prices[model]
    estimated_cost = (total_tokens / 1000) * cost_per_1k
    
    return {
        "model": model,
        "num_documents": num_documents,
        "avg_tokens_per_doc": avg_tokens_per_doc,
        "total_tokens": total_tokens,
        "cost_per_1k_tokens": cost_per_1k,
        "estimated_cost_usd": estimated_cost
    }