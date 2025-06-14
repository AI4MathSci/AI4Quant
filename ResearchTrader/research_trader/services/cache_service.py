"""
Caching service for ResearchTrader Paper objects.
"""

import time
from asyncio import Lock
from typing import Any

from research_trader.config import settings
from research_trader.models.paper import Paper  # Import the consolidated Paper model


class _MemoryCacheStore:
    """Internal in-memory cache implementation with TTL support."""

    def __init__(self, ttl: int = settings.CACHE_TTL):
        self._cache: dict[str, dict[str, Any]] = {}
        self._ttl = ttl
        self._lock = Lock()

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """Check if a cache entry has expired."""
        return time.time() >= entry["expiry"]

    async def get(self, key: str) -> Any | None:
        """
        Get a value from the cache store.

        Args:
            key: The cache key.

        Returns:
            Cached value or None if not found or expired.
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry:
                if not self._is_expired(entry):
                    return entry["value"]
                else:
                    # Remove expired entry
                    del self._cache[key]
        return None

    async def get_all(self) -> list[Any]:
        """Get all non-expired values from the cache store."""
        all_values = []
        # Iterate over a copy of keys to allow deletion during iteration
        keys = list(self._cache.keys())
        async with self._lock:
            for key in keys:
                entry = self._cache.get(key)  # Re-get in case it was deleted
                if entry:
                    if not self._is_expired(entry):
                        all_values.append(entry["value"])
                    else:
                        # Remove expired entry
                        del self._cache[key]
        return all_values

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set a value in the cache store.

        Args:
            key: The cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds (uses default if None).
        """
        expiry_time = time.time() + (ttl if ttl is not None else self._ttl)
        async with self._lock:
            self._cache[key] = {"value": value, "expiry": expiry_time}

    async def delete(self, key: str) -> bool:
        """
        Delete a value from the cache store.

        Args:
            key: The cache key.

        Returns:
            True if entry was found and deleted, False otherwise.
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
        return False

    async def clear(self) -> None:
        """Clear all cached values."""
        async with self._lock:
            self._cache.clear()


# Singleton instance of the memory cache store
_cache_store = _MemoryCacheStore()


class CacheService:
    """Service for caching and retrieving Paper objects."""

    @staticmethod
    async def get_paper(paper_id: str) -> Paper | None:
        """Get a cached Paper object by its ID."""
        if not settings.ENABLE_CACHE:
            return None

        cached_data = await _cache_store.get(paper_id)
        if cached_data:
            # Ensure data is valid Paper model (handles potential evolution)
            try:
                return Paper.model_validate(cached_data)
            except Exception:  # Catch validation errors
                # Log error or handle invalid cache data (e.g., delete it)
                await _cache_store.delete(paper_id)
                return None
        return None

    @staticmethod
    async def set_paper(paper: Paper, ttl: int | None = None) -> None:
        """Cache a Paper object."""
        if not settings.ENABLE_CACHE:
            return
        # Store the paper as a dictionary for broader compatibility
        await _cache_store.set(paper.paper_id, paper.model_dump(), ttl=ttl)

    @staticmethod
    async def get_all_papers() -> list[Paper]:
        """Get all cached Paper objects."""
        if not settings.ENABLE_CACHE:
            return []

        all_cached_data = await _cache_store.get_all()
        papers = []
        for data in all_cached_data:
            try:
                papers.append(Paper.model_validate(data))
            except Exception:
                # Log error or handle invalid cache data
                # For simplicity here, we just skip invalid entries
                pass
        return papers

    @staticmethod
    async def delete_paper(paper_id: str) -> bool:
        """Delete a cached Paper object by its ID."""
        if not settings.ENABLE_CACHE:
            return False
        return await _cache_store.delete(paper_id)

    @staticmethod
    async def clear_all() -> None:
        """Clear the entire paper cache."""
        if not settings.ENABLE_CACHE:
            return
        await _cache_store.clear()


# Optional: provide easy access to the service instance if needed elsewhere
# cache_service = CacheService()
