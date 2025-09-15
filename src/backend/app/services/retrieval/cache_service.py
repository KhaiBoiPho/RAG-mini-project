# src/backend/app/services/retrieval/cache_service.py
from typing import Any, Optional, Dict
import asyncio
import hashlib
import json
import time
from ...config import settings
from ...utils.logger import get_logger

logger = get_logger(__name__)

class CacheService:
    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = settings.cache_ttl
        self.max_size = 1000  # Maximum number of cached items
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_expired())
    
    def _generate_key(self, query: str) -> str:
        """Generate cache key from query"""
        return hashlib.md5(query.encode('utf-8')).hexdigest()
    
    async def get(self, query: str) -> Optional[Any]:
        """Get cached result for query"""
        try:
            key = self._generate_key(query)
            
            if key in self.cache:
                cached_item = self.cache[key]
                
                # Check if expired
                if time.time() - cached_item['timestamp'] > self.ttl:
                    del self.cache[key]
                    return None
                
                # Update access time for LRU
                cached_item['last_access'] = time.time()
                logger.debug(f"Cache hit for key: {key}")
                return cached_item['data']
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            return None
    
    async def set(self, query: str, data: Any) -> bool:
        """Set cached result for query"""
        try:
            key = self._generate_key(query)
            
            # Check cache size and evict if necessary
            if len(self.cache) >= self.max_size:
                await self._evict_lru()
            
            self.cache[key] = {
                'data': data,
                'timestamp': time.time(),
                'last_access': time.time()
            }
            
            logger.debug(f"Cache set for key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
            return False
    
    async def delete(self, query: str) -> bool:
        """Delete cached result for query"""
        try:
            key = self._generate_key(query)
            
            if key in self.cache:
                del self.cache[key]
                logger.debug(f"Cache deleted for key: {key}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            self.cache.clear()
            logger.info("Cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    async def _evict_lru(self):
        """Evict least recently used item"""
        if not self.cache:
            return
        
        # Find least recently used item
        lru_key = min(
            self.cache.keys(),
            key=lambda k: self.cache[k]['last_access']
        )
        
        del self.cache[lru_key]
        logger.debug(f"Evicted LRU item: {lru_key}")
    
    async def _cleanup_expired(self):
        """Periodically cleanup expired items"""
        while True:
            try:
                current_time = time.time()
                expired_keys = [
                    key for key, item in self.cache.items()
                    if current_time - item['timestamp'] > self.ttl
                ]
                
                for key in expired_keys:
                    del self.cache[key]
                
                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")
                
                # Run cleanup every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in cache cleanup: {str(e)}")
                await asyncio.sleep(60)  # Retry after 1 minute on error
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        
        expired_count = sum(
            1 for item in self.cache.values()
            if current_time - item['timestamp'] > self.ttl
        )
        
        return {
            "total_items": len(self.cache),
            "expired_items": expired_count,
            "active_items": len(self.cache) - expired_count,
            "max_size": self.max_size,
            "ttl": self.ttl
        }