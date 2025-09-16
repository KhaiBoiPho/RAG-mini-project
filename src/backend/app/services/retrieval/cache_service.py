# src/backend/app/services/retrieval/cache_service.py
from typing import Any, Optional, Dict, List, Union
import asyncio
import hashlib
import json
import time
from datetime import datetime, timedelta
from src.backend.app.config import settings
from src.backend.app.utils.logger import get_logger

logger = get_logger(__name__)

class CacheService:
    def __init__(self):
        # Configuration from settings
        self.ttl = getattr(settings, 'CACHE_TTL', 1800)  # 30 minutes default
        self.max_size = getattr(settings, 'CACHE_MAX_SIZE', 1000)
        self.enable_cache = getattr(settings, 'ENABLE_CACHE', True)
        
        # Cache storage
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_counts: Dict[str, int] = {}
        
        # Cleanup task
        self._cleanup_task = None
        # if self.enable_cache:
        #     self._start_cleanup_task()
        
        logger.info(f"Cache service initialized - TTL: {self.ttl}s, Max size: {self.max_size}")
    
    async def start(self):
        """Start cleanup in an async context"""
        if self.enable_cache and self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_expired())
            logger.info("Cache cleanup task started")
        
    def _generate_cache_key(
        self, 
        query: str, 
        method: str = "hybrid",
        filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> str:
        """Generate cache key from query parameters"""
        # Create a consistent key from all parameters
        cache_data = {
            "query": query.lower().strip(),
            "method": method,
            "filters": filters or {},
            "top_k": top_k
        }
        
        # Convert to JSON string for consistent hashing
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode('utf-8')).hexdigest()[:16]
    
    async def get_cached_results(
        self, 
        query: str, 
        method: str = "hybrid",
        filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results"""
        if not self.enable_cache:
            return None
            
        try:
            key = self._generate_cache_key(query, method, filters, top_k)
            
            if key in self.cache:
                cached_item = self.cache[key]
                
                # Check if expired
                if time.time() - cached_item['timestamp'] > self.ttl:
                    await self._remove_cache_entry(key)
                    logger.debug(f"Cache expired for key: {key}")
                    return None
                
                # Update access statistics
                cached_item['last_access'] = time.time()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                
                logger.info(f"Cache hit for query: '{query[:50]}...' (key: {key})")
                return cached_item['data']
            
            logger.debug(f"Cache miss for query: '{query[:50]}...'")
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            return None
    
    async def set_cached_results(
        self, 
        query: str, 
        results: List[Dict[str, Any]],
        method: str = "hybrid",
        filters: Optional[Dict] = None,
        top_k: int = 5
    ) -> bool:
        """Cache search results"""
        if not self.enable_cache or not results:
            return False
            
        try:
            key = self._generate_cache_key(query, method, filters, top_k)
            
            # Check cache size and evict if necessary
            if len(self.cache) >= self.max_size:
                await self._evict_entries()
            
            # Store in cache
            self.cache[key] = {
                'data': results,
                'timestamp': time.time(),
                'last_access': time.time(),
                'query': query[:100],  # Store truncated query for debugging
                'method': method,
                'result_count': len(results)
            }
            
            self.access_counts[key] = 1
            
            logger.info(f"Cached {len(results)} results for query: '{query[:50]}...' (key: {key})")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
            return False
    
    async def invalidate_query(self, query: str) -> int:
        """Invalidate all cache entries for a specific query"""
        try:
            invalidated_count = 0
            keys_to_remove = []
            
            # Find all keys that match this query
            for key, item in self.cache.items():
                if item.get('query', '').lower() == query.lower()[:100]:
                    keys_to_remove.append(key)
            
            # Remove found keys
            for key in keys_to_remove:
                await self._remove_cache_entry(key)
                invalidated_count += 1
            
            if invalidated_count > 0:
                logger.info(f"Invalidated {invalidated_count} cache entries for query: '{query[:50]}...'")
            
            return invalidated_count
            
        except Exception as e:
            logger.error(f"Error invalidating cache for query: {str(e)}")
            return 0
    
    async def clear_all(self) -> bool:
        """Clear all cache entries"""
        try:
            cleared_count = len(self.cache)
            self.cache.clear()
            self.access_counts.clear()
            
            logger.info(f"Cleared all cache entries ({cleared_count} items)")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    async def _remove_cache_entry(self, key: str):
        """Remove a single cache entry"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_counts:
            del self.access_counts[key]
    
    async def _evict_entries(self, count: int = 1):
        """Evict least recently used entries"""
        if not self.cache:
            return
        
        # Sort by last access time (LRU)
        sorted_keys = sorted(
            self.cache.keys(),
            key=lambda k: self.cache[k]['last_access']
        )
        
        # Evict oldest entries
        for i in range(min(count, len(sorted_keys))):
            key = sorted_keys[i]
            await self._remove_cache_entry(key)
            logger.debug(f"Evicted LRU cache entry: {key}")
    
    async def _cleanup_expired(self):
        """Periodically cleanup expired items"""
        cleanup_interval = min(300, self.ttl // 2)  # Run cleanup every 5 min or half TTL
        
        while True:
            try:
                if not self.enable_cache:
                    break
                    
                current_time = time.time()
                expired_keys = []
                
                # Find expired entries
                for key, item in self.cache.items():
                    if current_time - item['timestamp'] > self.ttl:
                        expired_keys.append(key)
                
                # Remove expired entries
                for key in expired_keys:
                    await self._remove_cache_entry(key)
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                
                # Sleep until next cleanup
                await asyncio.sleep(cleanup_interval)
                
            except asyncio.CancelledError:
                logger.info("Cache cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cache cleanup: {str(e)}")
                await asyncio.sleep(60)  # Retry after 1 minute on error
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if not self.enable_cache:
            return {"enabled": False}
        
        current_time = time.time()
        
        expired_count = 0
        total_hits = 0
        avg_result_count = 0
        method_breakdown = {}
        
        if self.cache:
            for key, item in self.cache.items():
                # Count expired items
                if current_time - item['timestamp'] > self.ttl:
                    expired_count += 1
                
                # Count access hits
                total_hits += self.access_counts.get(key, 0)
                
                # Result count stats
                avg_result_count += item.get('result_count', 0)
                
                # Method breakdown
                method = item.get('method', 'unknown')
                method_breakdown[method] = method_breakdown.get(method, 0) + 1
            
            if self.cache:
                avg_result_count = avg_result_count / len(self.cache)
        
        return {
            "enabled": True,
            "total_items": len(self.cache),
            "expired_items": expired_count,
            "active_items": len(self.cache) - expired_count,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl,
            "total_cache_hits": total_hits,
            "average_results_per_query": round(avg_result_count, 1),
            "method_breakdown": method_breakdown,
            "memory_usage_estimate": f"{len(self.cache) * 0.001:.2f} MB"  # Rough estimate
        }
    
    async def warm_cache(self, queries: List[str], retriever_func):
        """Pre-warm cache with common queries"""
        if not self.enable_cache:
            return
        
        logger.info(f"Warming cache with {len(queries)} queries...")
        
        for query in queries:
            try:
                # Check if already cached
                cached = await self.get_cached_results(query)
                if cached is None:
                    # Retrieve and cache
                    results, _ = await retriever_func(query)
                    await self.set_cached_results(query, results)
                    
            except Exception as e:
                logger.warning(f"Failed to warm cache for query '{query[:30]}...': {str(e)}")
        
        logger.info("Cache warming completed")
    
    def disable_cache(self):
        """Temporarily disable caching"""
        self.enable_cache = False
        logger.info("Cache disabled")
    
    def enable_cache_service(self):
        """Re-enable caching"""
        self.enable_cache = True
        if self._cleanup_task is None:
            self._start_cleanup_task()
        logger.info("Cache enabled")
    
    async def shutdown(self):
        """Cleanup on shutdown"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.clear_all()
        logger.info("Cache service shut down")