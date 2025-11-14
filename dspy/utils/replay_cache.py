"""ReplayCache: A deterministic cache for testing DSPy programs.

This module provides a context manager for recording and replaying LM requests
in a deterministic way, enabling fast, offline, and cost-free testing.
"""

import copy
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pydantic

logger = logging.getLogger(__name__)


class CacheMissError(Exception):
    """Raised when replay mode encounters a cache miss."""

    def __init__(self, request: dict, cache_key: str, cache_file: str):
        """Initialize CacheMissError with detailed information.

        Args:
            request: The request that caused the cache miss
            cache_key: The computed cache key
            cache_file: Path to the cache file
        """
        self.request = request
        self.cache_key = cache_key
        self.cache_file = cache_file

        # Extract useful info from request for error message
        model = request.get("model", "unknown")
        fn_identifier = request.get("_fn_identifier", "unknown")

        message = (
            f"Cache miss in replay mode!\n"
            f"  Cache file: {cache_file}\n"
            f"  Cache key: {cache_key}\n"
            f"  Request model: {model}\n"
            f"  Request function: {fn_identifier}\n"
            f"\n"
            f"To record this request, run with:\n"
            f"  CACHE_MODE=record python <your_test>.py\n"
        )
        super().__init__(message)


class ReplayCache:
    """A deterministic cache for testing that records and replays LM requests.

    ReplayCache provides two modes of operation:
    - **replay** (default): Only uses recorded responses, never calls the LM API
    - **record**: Delegates to underlying cache and records all requests

    The cache uses JSON files for storage, making them human-readable and
    version-control friendly. Unused cache entries are automatically cleaned
    up to keep files minimal.

    Examples:
        Basic usage in replay mode (default):
        ```python
        with ReplayCache("tests/fixtures/my_test.json"):
            result = my_program(input)  # Uses recorded responses
        ```

        Recording new responses:
        ```python
        with ReplayCache("tests/fixtures/my_test.json", mode="record"):
            result = my_program(input)  # Records new responses
        ```

        Using environment variable:
        ```bash
        CACHE_MODE=record pytest tests/my_test.py
        ```

    Attributes:
        path: Path to the JSON cache file
        mode: Current mode ("record" or "replay")
    """

    def __init__(self, path: str, mode: str | None = None):
        """Initialize ReplayCache.

        Args:
            path: Path to JSON file for cache storage. Will be created if it
                doesn't exist in record mode.
            mode: "record" or "replay". If None, uses CACHE_MODE environment
                variable, defaults to "replay".

        Raises:
            ValueError: If mode is invalid
        """
        self.path = Path(path)
        self.mode = self._determine_mode(mode)

        # Internal state
        self._original_cache = None
        self._delegate = None
        self._cache_data = {}  # Maps cache_key -> {"request": ..., "response": ...}
        self._accessed_keys = set()  # Track which keys were accessed for cleanup

    def _determine_mode(self, mode: str | None) -> str:
        """Determine the cache mode.

        Priority order:
        1. Explicit mode parameter
        2. CACHE_MODE environment variable
        3. Default to "replay"

        Args:
            mode: Explicit mode or None

        Returns:
            The determined mode ("record" or "replay")

        Raises:
            ValueError: If mode is invalid
        """
        if mode is not None:
            mode = mode.lower()
            if mode not in ("record", "replay"):
                raise ValueError(f"Invalid mode: {mode}. Must be 'record' or 'replay'")
            return mode

        env_mode = os.environ.get("CACHE_MODE", "replay").lower()
        if env_mode not in ("record", "replay"):
            raise ValueError(f"Invalid CACHE_MODE: {env_mode}. Must be 'record' or 'replay'")

        return env_mode

    def __enter__(self) -> "ReplayCache":
        """Enter the context manager.

        - Loads cache from JSON file
        - Saves reference to current dspy.cache
        - Sets self as the global cache
        - In record mode, sets up delegate cache

        Returns:
            self

        Raises:
            FileNotFoundError: In replay mode, if cache file doesn't exist
            json.JSONDecodeError: If cache file is corrupted
        """
        import dspy

        # Save original cache
        self._original_cache = dspy.cache

        # Load cache data
        self._load_cache()

        if self.mode == "record":
            # In record mode, delegate to the original cache
            self._delegate = self._original_cache
        else:
            # In replay mode, no delegate
            self._delegate = None

        # Initialize accessed keys tracker
        self._accessed_keys = set()

        # Set self as the global cache
        dspy.cache = self

        logger.debug(f"ReplayCache entered: mode={self.mode}, path={self.path}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the context manager.

        - In record mode, saves cache data to JSON file (with cleanup)
        - Restores original cache
        - Does not suppress exceptions

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)

        Returns:
            None (does not suppress exceptions)
        """
        import dspy

        # In record mode, save the cache with cleanup
        if self.mode == "record":
            self._save_cache()

        # Restore original cache
        dspy.cache = self._original_cache

        logger.debug(f"ReplayCache exited: mode={self.mode}, path={self.path}")

        # Don't suppress exceptions
        return None

    def get(self, request: dict, ignored_args_for_cache_key: list[str] | None = None) -> Any:
        """Get a cached response.

        In record mode: Delegates to underlying cache
        In replay mode: Returns cached response or raises CacheMissError

        Args:
            request: The request dictionary
            ignored_args_for_cache_key: Arguments to ignore when computing cache key

        Returns:
            The cached response (or None if not found in record mode)

        Raises:
            CacheMissError: In replay mode, if request is not in cache
        """
        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception as e:
            logger.debug(f"Failed to generate cache key: {e}")
            if self.mode == "record" and self._delegate:
                return self._delegate.get(request, ignored_args_for_cache_key)
            return None

        # Track that this key was accessed (for cleanup)
        self._accessed_keys.add(key)

        if self.mode == "record":
            # In record mode, delegate to underlying cache
            result = self._delegate.get(request, ignored_args_for_cache_key) if self._delegate else None

            # Store in our cache data if we got a result
            if result is not None:
                self._cache_data[key] = {
                    "request": copy.deepcopy(request),
                    "response": result,
                }

            return result
        else:
            # In replay mode, only use cached data
            if key in self._cache_data:
                # Deserialize and return
                entry = self._cache_data[key]
                response = self._deserialize_value(entry["response"])

                # Deep copy to avoid mutation
                return copy.deepcopy(response)
            else:
                # Cache miss in replay mode - this is an error
                raise CacheMissError(request, key, str(self.path))

    def put(
        self,
        request: dict,
        value: Any,
        ignored_args_for_cache_key: list[str] | None = None,
        enable_memory_cache: bool = True,
    ) -> None:
        """Store a response in the cache.

        In record mode: Delegates to underlying cache and records locally
        In replay mode: Logs a warning (should not be called)

        Args:
            request: The request dictionary
            value: The response value to cache
            ignored_args_for_cache_key: Arguments to ignore when computing cache key
            enable_memory_cache: Whether to enable memory cache (passed to delegate)
        """
        try:
            key = self.cache_key(request, ignored_args_for_cache_key)
        except Exception as e:
            logger.debug(f"Failed to generate cache key: {e}")
            if self.mode == "record" and self._delegate:
                self._delegate.put(request, value, ignored_args_for_cache_key, enable_memory_cache)
            return

        # Track that this key was accessed
        self._accessed_keys.add(key)

        if self.mode == "record":
            # Delegate to underlying cache
            if self._delegate:
                self._delegate.put(request, value, ignored_args_for_cache_key, enable_memory_cache)

            # Store in our cache data
            self._cache_data[key] = {
                "request": copy.deepcopy(request),
                "response": value,
            }
        else:
            # In replay mode, putting to cache is unexpected but not an error
            logger.warning(f"put() called in replay mode (key={key}). This should not happen.")

    def cache_key(self, request: dict, ignored_args_for_cache_key: list[str] | None = None) -> str:
        """Generate a cache key for a request.

        Delegates to the Cache.cache_key implementation for consistency.

        Args:
            request: The request dictionary
            ignored_args_for_cache_key: Arguments to ignore when computing cache key

        Returns:
            A hash string representing the cache key
        """
        from dspy.clients.cache import Cache

        # Use a temporary Cache instance to compute the key
        # This ensures consistency with DSPy's cache key generation
        temp_cache = Cache(
            enable_disk_cache=False,
            enable_memory_cache=False,
            disk_cache_dir="",
            memory_max_entries=0,
        )
        return temp_cache.cache_key(request, ignored_args_for_cache_key)

    def _load_cache(self) -> None:
        """Load cache from JSON file.

        In record mode: Loads existing file or starts with empty cache
        In replay mode: Must load existing file or raises error

        Raises:
            FileNotFoundError: In replay mode, if file doesn't exist
            json.JSONDecodeError: If file is corrupted
        """
        if not self.path.exists():
            if self.mode == "replay":
                raise FileNotFoundError(
                    f"Cache file not found: {self.path}\n"
                    f"In replay mode, the cache file must exist.\n"
                    f"To create it, run in record mode:\n"
                    f"  CACHE_MODE=record python <your_test>.py"
                )
            else:
                # In record mode, start with empty cache
                self._cache_data = {}
                logger.debug(f"Cache file doesn't exist, starting with empty cache: {self.path}")
                return

        # Load from file
        try:
            with open(self.path) as f:
                data = json.load(f)

            # Extract entries
            self._cache_data = data.get("entries", {})

            logger.debug(f"Loaded cache with {len(self._cache_data)} entries from {self.path}")

        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse cache file: {self.path}. "
                f"The file may be corrupted. Error: {e.msg}",
                e.doc,
                e.pos,
            )

    def _save_cache(self) -> None:
        """Save cache to JSON file.

        Only saves entries that were accessed during this session (cleanup).
        Creates parent directories if needed.
        """
        # Cleanup: only keep entries that were accessed
        cleaned_entries = {k: v for k, v in self._cache_data.items() if k in self._accessed_keys}

        # Serialize all responses
        serialized_entries = {}
        for key, entry in cleaned_entries.items():
            try:
                serialized_entries[key] = {
                    "request": entry["request"],
                    "response": self._serialize_value(entry["response"]),
                }
            except Exception as e:
                logger.warning(f"Failed to serialize cache entry (key={key}): {e}")
                continue

        # Build full cache structure
        cache_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "mode": self.mode,
            },
            "entries": serialized_entries,
        }

        # Try to add DSPy version
        try:
            import dspy

            cache_data["metadata"]["dspy_version"] = dspy.__version__
        except Exception:
            pass

        # Create parent directories if needed
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(self.path, "w") as f:
            json.dump(cache_data, f, indent=2, sort_keys=True)

        logger.debug(
            f"Saved cache with {len(serialized_entries)} entries "
            f"(cleaned from {len(self._cache_data)}) to {self.path}"
        )

    def _serialize_value(self, value: Any) -> dict:
        """Serialize a response value to JSON-compatible format.

        Args:
            value: The value to serialize

        Returns:
            A dictionary with "type" and data fields

        Raises:
            ValueError: If value cannot be serialized
        """
        # Import here to avoid circular dependencies
        from dspy.primitives.prediction import Prediction

        if isinstance(value, Prediction):
            return {
                "type": "Prediction",
                "store": dict(value._store),
                "usage": getattr(value, "_lm_usage", {}),
                "cache_hit": getattr(value, "cache_hit", False),
            }
        elif isinstance(value, dict):
            return {"type": "dict", "data": value}
        elif isinstance(value, pydantic.BaseModel):
            return {
                "type": "pydantic",
                "class": f"{value.__class__.__module__}.{value.__class__.__name__}",
                "data": value.model_dump(mode="json"),
            }
        elif value is None or isinstance(value, str | int | float | bool):
            return {"type": "primitive", "data": value}
        elif isinstance(value, list):
            return {"type": "list", "data": value}
        else:
            # Fallback: try to serialize as-is
            try:
                # Test if it's JSON serializable
                json.dumps(value)
                return {"type": "json", "data": value}
            except (TypeError, ValueError) as e:
                raise ValueError(f"Cannot serialize value of type {type(value).__name__}: {e}")

    def _deserialize_value(self, data: dict) -> Any:
        """Deserialize a response value from JSON format.

        Args:
            data: The serialized data dictionary

        Returns:
            The deserialized value

        Raises:
            ValueError: If type is unknown
        """
        value_type = data.get("type")

        if value_type == "Prediction":
            from dspy.primitives.prediction import Prediction

            pred = Prediction()
            pred._store = data["store"]
            pred._lm_usage = data.get("usage", {})
            if hasattr(pred, "cache_hit"):
                pred.cache_hit = data.get("cache_hit", False)
            return pred
        elif value_type == "dict":
            return data["data"]
        elif value_type == "pydantic":
            # For now, just return as dict
            # Future enhancement: reconstruct the actual pydantic model
            return data["data"]
        elif value_type == "primitive":
            return data["data"]
        elif value_type == "list":
            return data["data"]
        elif value_type == "json":
            return data["data"]
        else:
            raise ValueError(f"Unknown serialization type: {value_type}")
