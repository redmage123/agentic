# services/generative_agent/infrastructure/memo_manager.py
from typing import Dict, Any, Optional, Callable, Tuple
import time
import asyncio
from datetime import datetime
import hashlib
import json

class AsyncMemoizer:
   def __init__(self, ttl: int = 3600):
       self.cache: Dict[str, Dict[str, Any]] = {}
       self.ttl = ttl
       self._lock = asyncio.Lock()

   async def get(self, key: str) -> Optional[Any]:
       if key not in self.cache:
           return None
       entry = self.cache[key]
       if time.time() > entry['expires_at']:
           del self.cache[key]
           return None
       return entry['value']

   async def set(self, key: str, value: Any):
       self.cache[key] = {
           'value': value,
           'expires_at': time.time() + self.ttl
       }

class MemoManager:
   def __init__(self):
       self.caches: Dict[str, AsyncMemoizer] = {}
       self._lock = asyncio.Lock()
       self._stats: Dict[str, Dict[str, Any]] = {}

   async def get_memoizer(self, name: str, ttl: int = 3600) -> AsyncMemoizer:
       async with self._lock:
           if name not in self.caches:
               self.caches[name] = AsyncMemoizer(ttl=ttl)
               self._stats[name] = {
                   'created_at': datetime.now(),
                   'hits': 0,
                   'misses': 0,
                   'total_calls': 0
               }
           return self.caches[name]

   async def clear_cache(self, name: str) -> None:
       async with self._lock:
           if name in self.caches:
               self.caches[name].cache.clear()

   async def clear_all(self) -> None:
       async with self._lock:
           for cache in self.caches.values():
               cache.cache.clear()

   def get_stats(self) -> Dict[str, Dict[str, Any]]:
       return {
           name: {
               **stats,
               'size': len(self.caches[name].cache),
               'hit_rate': stats['hits'] / stats['total_calls'] if stats['total_calls'] > 0 else 0
           }
           for name, stats in self._stats.items()
       }

   async def cleanup_expired(self) -> Dict[str, int]:
       cleaned = {}
       async with self._lock:
           for name, cache in self.caches.items():
               count = 0
               current_time = time.time()
               keys_to_delete = [
                   k for k, v in cache.cache.items() 
                   if current_time > v['expires_at']
               ]
               for key in keys_to_delete:
                   del cache.cache[key]
                   count += 1
               cleaned[name] = count
       return cleaned

# Global memo manager instance
memo_manager = MemoManager()

def make_key(func: Callable, args: Tuple, kwargs: Dict) -> str:
   key_parts = [
       func.__name__,
       str(args),
       str(sorted(kwargs.items()))
   ]
   key_str = "|".join(key_parts)
   return hashlib.sha256(key_str.encode()).hexdigest()

def managed_memoize(cache_name: str, ttl: int = 3600):
   def decorator(func):
       async def wrapper(*args, **kwargs):
           memoizer = await memo_manager.get_memoizer(cache_name, ttl)
           async with memoizer._lock:
               key = make_key(func, args, kwargs)
               memo_manager._stats[cache_name]['total_calls'] += 1
               
               result = await memoizer.get(key)
               if result is not None:
                   memo_manager._stats[cache_name]['hits'] += 1
                   return result
               
               memo_manager._stats[cache_name]['misses'] += 1
               result = await func(*args, **kwargs)
               await memoizer.set(key, result)
               return result
       return wrapper
   return decorator

# Usage examples in other classes:
class FileSystemPromptManager(PromptManager):
   @managed_memoize(cache_name="prompt_loading", ttl=3600)
   async def load_prompt(self, name: str) -> str:
       # Implementation...
       pass

   @managed_memoize(cache_name="prompt_metadata", ttl=3600)
   async def get_prompt_metadata(self, name: str) -> PromptMetadata:
       # Implementation...
       pass

class GenerativeService:
   @managed_memoize(cache_name="pattern_analysis", ttl=1800)
   async def analyze_pattern(self, data: Dict[str, Any]) -> AnalysisResponse:
       # Implementation...
       pass

   @managed_memoize(cache_name="prompt_chains", ttl=1800)
   async def compose_prompt_chain(
       self,
       analysis_type: AnalysisType,
       context: Dict[str, Any]
   ) -> PromptChain:
       # Implementation...
       pass

class AnthropicClient:
   @managed_memoize(cache_name="token_counting", ttl=300)
   async def count_tokens(self, text: str) -> int:
       # Implementation...
       pass
