# Ultra-Detailed Plan: ReplayCache for DSPy

**Date**: 2025-11-14
**Version**: 1.0
**Author**: Claude (via GitHub Issue)

## 1. Architecture Overview

### 1.1 Core Design Principles
- **Simple & Clean**: Minimal API surface, easy to understand
- **Context Manager**: Natural Python idiom for temporary cache swapping
- **Two-Mode Operation**: Record and Replay modes with clear separation
- **Cache Delegation**: Wraps existing cache in record mode
- **JSON-Based Storage**: Human-readable, version-control friendly
- **Automatic Cleanup**: Unused cache entries are removed

### 1.2 Cache Interface Compatibility
The ReplayCache must implement the DSPy cache interface:
- `get(request: dict, ignored_args_for_cache_key: list[str] | None) -> Any`
- `put(request: dict, value: Any, ignored_args_for_cache_key: list[str] | None, enable_memory_cache: bool) -> None`
- `cache_key(request: dict, ignored_args_for_cache_key: list[str] | None) -> str`

### 1.3 File Location
- Implementation: `dspy/utils/replay_cache.py`
- Tests: `tests/utils/test_replay_cache.py`
- Export from: `dspy/__init__.py` and `dspy/utils/__init__.py`

## 2. Detailed Design

### 2.1 Class Structure

```python
class ReplayCache:
    """
    A deterministic cache for testing that records and replays LM requests.

    Modes:
        - replay (default): Only uses recorded responses, never calls delegate
        - record: Delegates to underlying cache and records all requests

    Usage:
        # Replay mode (default)
        with ReplayCache("tests/fixtures/my_test.json"):
            result = my_program(input)  # Uses recorded responses

        # Record mode
        with ReplayCache("tests/fixtures/my_test.json", mode="record"):
            result = my_program(input)  # Records new responses

        # Or set via environment variable
        # CACHE_MODE=record python test.py
    """

    def __init__(self, path: str, mode: str | None = None):
        """
        Args:
            path: Path to JSON file for cache storage
            mode: "record" or "replay". If None, uses CACHE_MODE env var,
                  defaults to "replay"
        """

    def __enter__(self) -> "ReplayCache":
        """Enter context: load cache, save original, set self as global cache"""

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context: restore original cache, save if record mode"""

    def get(self, request: dict, ignored_args_for_cache_key: list[str] | None = None) -> Any:
        """Get cached response or raise/delegate depending on mode"""

    def put(self, request: dict, value: Any,
            ignored_args_for_cache_key: list[str] | None = None,
            enable_memory_cache: bool = True) -> None:
        """Store response (record mode only)"""

    def cache_key(self, request: dict,
                  ignored_args_for_cache_key: list[str] | None = None) -> str:
        """Delegate to underlying Cache.cache_key for consistency"""
```

### 2.2 Internal Methods

```python
    def _load_cache(self) -> dict:
        """Load cache from JSON file, return empty dict if not exists"""

    def _save_cache(self) -> None:
        """Save cache to JSON file, only including accessed entries"""

    def _serialize_value(self, value: Any) -> dict:
        """Serialize response to JSON-compatible format"""

    def _deserialize_value(self, data: dict) -> Any:
        """Deserialize response from JSON format"""
```

### 2.3 JSON File Structure

```json
{
  "metadata": {
    "created_at": "2025-11-14T12:00:00Z",
    "last_updated": "2025-11-14T12:05:00Z",
    "dspy_version": "3.0.4b2",
    "mode": "replay"
  },
  "entries": {
    "abc123def456...": {
      "request": {
        "model": "openai/gpt-4o-mini",
        "messages": [...],
        "temperature": 0.0,
        "_fn_identifier": "dspy.clients.lm.LM.__call__"
      },
      "response": {
        "type": "Prediction",
        "data": {
          "answer": "Paris",
          "reasoning": "France's capital is Paris"
        },
        "usage": {},
        "cache_hit": false
      }
    }
  }
}
```

**Key Design Decisions:**
- **metadata**: Track when cache was created, last updated, DSPy version
- **entries**: Map from hash to request+response pairs
- **request stored**: Enables debugging (see what's cached)
- **response.type**: Enables proper deserialization
- **Cleanup strategy**: Only save entries that were accessed during session

## 3. Implementation Details

### 3.1 Mode Detection Logic

```python
def _determine_mode(self, mode: str | None) -> str:
    """
    Priority order:
    1. Explicit mode parameter
    2. CACHE_MODE environment variable
    3. Default to "replay"
    """
    if mode is not None:
        if mode not in ("record", "replay"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'record' or 'replay'")
        return mode

    env_mode = os.environ.get("CACHE_MODE", "replay").lower()
    if env_mode not in ("record", "replay"):
        raise ValueError(f"Invalid CACHE_MODE: {env_mode}. Must be 'record' or 'replay'")

    return env_mode
```

### 3.2 Record Mode Flow

1. **Enter context**:
   - Load existing JSON (if exists)
   - Save reference to current `dspy.cache`
   - Set delegate cache
   - Set `dspy.cache = self`
   - Initialize `_accessed_keys = set()`

2. **get() call**:
   - Generate cache key
   - Mark key as accessed: `_accessed_keys.add(key)`
   - Delegate to underlying cache: `return self._delegate.get(request, ignored_args)`

3. **put() call**:
   - Generate cache key
   - Mark key as accessed
   - Delegate: `self._delegate.put(request, value, ignored_args, enable_memory_cache)`
   - Store in memory: `self._cache_data[key] = {"request": request, "response": value}`

4. **Exit context**:
   - Filter `_cache_data` to only include `_accessed_keys`
   - Save cleaned JSON
   - Restore `dspy.cache = self._original_cache`

### 3.3 Replay Mode Flow

1. **Enter context**:
   - Load JSON (must exist or raise clear error)
   - Save reference to current `dspy.cache`
   - Set delegate to None
   - Set `dspy.cache = self`

2. **get() call**:
   - Generate cache key
   - Lookup in loaded cache
   - If found: deserialize and return
   - If not found: raise `CacheMissError` with helpful message

3. **put() call**:
   - Ignore (should never be called in replay mode, but don't error)
   - Log warning if this happens

4. **Exit context**:
   - Restore `dspy.cache = self._original_cache`
   - No save needed

### 3.4 Serialization Strategy

**Challenge**: LM responses can be complex objects (Prediction, dict, dataclass, etc.)

**Solution**:
```python
def _serialize_value(self, value: Any) -> dict:
    """Serialize response to JSON format"""
    if isinstance(value, Prediction):
        return {
            "type": "Prediction",
            "store": dict(value._store),
            "usage": getattr(value, "usage", {}),
            "cache_hit": getattr(value, "cache_hit", False)
        }
    elif isinstance(value, dict):
        return {
            "type": "dict",
            "data": value
        }
    elif isinstance(value, pydantic.BaseModel):
        return {
            "type": "pydantic",
            "class": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "data": value.model_dump(mode="json")
        }
    else:
        # Fallback: try JSON serialization
        try:
            orjson.dumps(value)
            return {"type": "json", "data": value}
        except:
            raise ValueError(f"Cannot serialize value of type {type(value)}")

def _deserialize_value(self, data: dict) -> Any:
    """Deserialize response from JSON format"""
    value_type = data.get("type")

    if value_type == "Prediction":
        from dspy.primitives.prediction import Prediction
        pred = Prediction()
        pred._store = data["store"]
        pred.usage = data.get("usage", {})
        pred.cache_hit = data.get("cache_hit", False)
        return pred
    elif value_type == "dict":
        return data["data"]
    elif value_type == "pydantic":
        # For now, just return as dict
        # Future: support reconstructing pydantic models
        return data["data"]
    elif value_type == "json":
        return data["data"]
    else:
        raise ValueError(f"Unknown type: {value_type}")
```

### 3.5 Error Handling

**Custom Exception**:
```python
class CacheMissError(Exception):
    """Raised when replay mode encounters a cache miss"""

    def __init__(self, request: dict, cache_key: str, cache_file: str):
        self.request = request
        self.cache_key = cache_key
        self.cache_file = cache_file

        message = (
            f"Cache miss in replay mode!\n"
            f"  Cache file: {cache_file}\n"
            f"  Cache key: {cache_key}\n"
            f"  Request model: {request.get('model', 'unknown')}\n"
            f"  Request function: {request.get('_fn_identifier', 'unknown')}\n"
            f"\n"
            f"To record this request, run with:\n"
            f"  CACHE_MODE=record python <your_test>.py\n"
        )
        super().__init__(message)
```

### 3.6 Cleanup Strategy

**Goal**: Keep JSON files minimal by removing unused entries

**Implementation**:
```python
# In __enter__
self._accessed_keys = set()

# In get() and put()
key = self.cache_key(request, ignored_args_for_cache_key)
self._accessed_keys.add(key)

# In __exit__ (record mode only)
cleaned_entries = {
    k: v for k, v in self._cache_data.items()
    if k in self._accessed_keys
}
self._save_cache(cleaned_entries)
```

## 4. Testing Strategy

### 4.1 Test File Structure

```
tests/utils/test_replay_cache.py
├── test_initialization
├── test_context_manager_record_mode
├── test_context_manager_replay_mode
├── test_record_mode_basic
├── test_record_mode_creates_json
├── test_record_mode_updates_existing
├── test_record_mode_cleanup_unused
├── test_replay_mode_basic
├── test_replay_mode_cache_miss_error
├── test_replay_mode_never_calls_delegate
├── test_mode_detection_explicit
├── test_mode_detection_env_var
├── test_mode_detection_default
├── test_serialization_prediction
├── test_serialization_dict
├── test_serialization_pydantic
├── test_integration_with_dspy_lm
└── test_concurrent_contexts (advanced)
```

### 4.2 Key Test Scenarios

**Test 1: Record Mode Basic**
```python
def test_record_mode_basic(tmp_path):
    cache_file = tmp_path / "test.json"
    original_cache = dspy.cache

    # Create a delegate cache
    delegate = Cache(
        enable_disk_cache=False,
        enable_memory_cache=True,
        disk_cache_dir="",
        memory_max_entries=100
    )

    # Pre-populate delegate with a response
    request = {"model": "test", "prompt": "hello"}
    response = Prediction(answer="world")
    delegate.put(request, response)

    # Use ReplayCache in record mode
    with ReplayCache(str(cache_file), mode="record"):
        # Verify cache was swapped
        assert dspy.cache != original_cache
        assert isinstance(dspy.cache, ReplayCache)

        # Get from cache (should delegate)
        result = dspy.cache.get(request)
        assert result.answer == "world"

    # Verify cache was restored
    assert dspy.cache == original_cache

    # Verify JSON was created and contains entry
    assert cache_file.exists()
    with open(cache_file) as f:
        data = json.load(f)

    assert "entries" in data
    assert len(data["entries"]) == 1
```

**Test 2: Replay Mode Cache Miss**
```python
def test_replay_mode_cache_miss(tmp_path):
    cache_file = tmp_path / "test.json"

    # Create minimal cache file
    with open(cache_file, "w") as f:
        json.dump({"metadata": {}, "entries": {}}, f)

    with ReplayCache(str(cache_file), mode="replay"):
        request = {"model": "test", "prompt": "unknown"}

        # Should raise CacheMissError
        with pytest.raises(CacheMissError) as exc_info:
            dspy.cache.get(request)

        # Check error message contains helpful info
        assert "Cache miss" in str(exc_info.value)
        assert "CACHE_MODE=record" in str(exc_info.value)
```

**Test 3: Cleanup Unused Entries**
```python
def test_cleanup_unused_entries(tmp_path):
    cache_file = tmp_path / "test.json"

    # Create cache with multiple entries
    entries = {
        "key1": {"request": {"prompt": "1"}, "response": {"type": "dict", "data": {"a": 1}}},
        "key2": {"request": {"prompt": "2"}, "response": {"type": "dict", "data": {"a": 2}}},
        "key3": {"request": {"prompt": "3"}, "response": {"type": "dict", "data": {"a": 3}}},
    }
    with open(cache_file, "w") as f:
        json.dump({"metadata": {}, "entries": entries}, f)

    # Use cache but only access key1 and key2
    with ReplayCache(str(cache_file), mode="record"):
        dspy.cache.get({"prompt": "1"})
        dspy.cache.get({"prompt": "2"})
        # key3 is not accessed

    # Verify only accessed keys remain
    with open(cache_file) as f:
        data = json.load(f)

    assert len(data["entries"]) == 2
    assert "key3" not in data["entries"]
```

**Test 4: Integration with Real LM**
```python
@pytest.mark.llm_call
def test_integration_with_lm(tmp_path, lm_for_test):
    cache_file = tmp_path / "lm_test.json"

    # Record mode: make real LM call
    with ReplayCache(str(cache_file), mode="record"):
        dspy.configure(lm=dspy.LM(lm_for_test))

        predictor = dspy.Predict("question -> answer")
        result1 = predictor(question="What is 2+2?")
        answer1 = result1.answer

    # Replay mode: use recorded response
    with ReplayCache(str(cache_file), mode="replay"):
        # Don't configure LM - should work without it
        predictor = dspy.Predict("question -> answer")
        result2 = predictor(question="What is 2+2?")
        answer2 = result2.answer

    # Answers should match
    assert answer1 == answer2
```

### 4.3 Test Organization

1. **Unit Tests**: Test individual methods in isolation
2. **Integration Tests**: Test with real DSPy components (marked with `@pytest.mark.llm_call`)
3. **Edge Cases**: Empty files, corrupted JSON, permission errors
4. **Concurrent Access**: Multiple contexts (though not primary use case)

## 5. Implementation Steps

### Phase 1: Core Implementation
1. Create `dspy/utils/replay_cache.py`
2. Implement `ReplayCache` class with basic structure
3. Implement mode detection
4. Implement context manager (`__enter__`, `__exit__`)
5. Implement cache interface (`get`, `put`, `cache_key`)
6. Implement JSON load/save
7. Implement serialization/deserialization for Prediction and dict types

### Phase 2: Error Handling & Edge Cases
1. Create `CacheMissError` exception
2. Add validation for mode values
3. Handle missing cache files appropriately
4. Handle JSON parsing errors
5. Add logging for debugging

### Phase 3: Testing
1. Create `tests/utils/test_replay_cache.py`
2. Implement unit tests for all methods
3. Implement integration tests with DSPy components
4. Add tests for edge cases
5. Ensure 100% code coverage

### Phase 4: Documentation & Polish
1. Add comprehensive docstrings
2. Add usage examples in docstrings
3. Update `dspy/__init__.py` to export `ReplayCache`
4. Add entry to `CLAUDE.md` under "Testing" section
5. Consider adding a tutorial/example

### Phase 5: Advanced Features (Future)
1. Support for Image, Audio, and other modalities
2. Support for pydantic model reconstruction
3. Cache versioning/migration
4. Cache diff tool for debugging
5. Auto-cleanup of stale caches

## 6. Example Usage

### 6.1 Basic Test Usage

```python
# tests/my_module/test_qa.py
import dspy
from dspy import ReplayCache

def test_qa_system():
    """Test QA system with recorded responses"""
    with ReplayCache("tests/fixtures/qa_test.json"):
        # Configure your program
        qa = MyQASystem()

        # Run test - uses recorded responses
        result = qa(question="What is DSPy?")
        assert "framework" in result.answer.lower()
```

### 6.2 Recording New Responses

```bash
# Record new responses
CACHE_MODE=record pytest tests/my_module/test_qa.py

# Commit the updated cache file
git add tests/fixtures/qa_test.json
git commit -m "Update QA test fixtures"
```

### 6.3 Multiple Test Cases

```python
def test_multiple_queries():
    """Test multiple queries with same cache file"""
    with ReplayCache("tests/fixtures/multi_qa.json"):
        qa = MyQASystem()

        # All these use recorded responses
        r1 = qa(question="What is DSPy?")
        r2 = qa(question="How does optimization work?")
        r3 = qa(question="What are signatures?")

        assert all([r1.answer, r2.answer, r3.answer])
```

## 7. Benefits

### 7.1 For Testing
- **Deterministic**: Same inputs always produce same outputs
- **Fast**: No LM API calls during tests
- **Offline**: Tests work without internet
- **Cost**: No API costs for CI/CD

### 7.2 For Development
- **Debugging**: Inspect exact requests/responses in JSON
- **Version Control**: Cache files track expected behavior
- **CI/CD**: Fast, reliable tests in pipelines
- **Reproducibility**: Share exact test scenarios

### 7.3 For Collaboration
- **Reviewable**: Cache files show what changed
- **Portable**: Share test fixtures across team
- **Documented**: JSON shows expected behavior

## 8. Future Enhancements

### 8.1 Multi-Modal Support
```python
# After initial implementation
def _serialize_value(self, value: Any) -> dict:
    if isinstance(value, Image):
        return {
            "type": "Image",
            "data": base64.b64encode(value.bytes).decode(),
            "format": value.format
        }
    # ... other types
```

### 8.2 Cache Validation
```python
# Validate cache on load
def _validate_cache(self, data: dict) -> None:
    """Ensure cache format is valid"""
    if data.get("metadata", {}).get("dspy_version") != dspy.__version__:
        logger.warning("Cache created with different DSPy version")
```

### 8.3 Cache Diff Tool
```python
# CLI tool to compare caches
def diff_caches(old_path: str, new_path: str) -> None:
    """Show differences between two cache files"""
    # Useful for debugging test changes
```

## 9. Summary

This implementation provides:
- ✅ **Simple API**: Context manager with clear modes
- ✅ **Deterministic**: Replay mode never calls LM
- ✅ **Clean**: Auto-cleanup of unused entries
- ✅ **Debuggable**: Human-readable JSON format
- ✅ **Well-tested**: Comprehensive test coverage
- ✅ **Well-designed**: Follows DSPy conventions
- ✅ **Extensible**: Ready for multi-modal support

The design is intentionally minimal for v1, with clear paths for enhancement. The focus is on correctness, simplicity, and excellent test coverage.

## 10. API Reference

### ReplayCache

```python
class ReplayCache:
    def __init__(self, path: str, mode: str | None = None)
    def __enter__(self) -> "ReplayCache"
    def __exit__(self, exc_type, exc_val, exc_tb) -> None
    def get(self, request: dict, ignored_args_for_cache_key: list[str] | None = None) -> Any
    def put(self, request: dict, value: Any, ignored_args_for_cache_key: list[str] | None = None, enable_memory_cache: bool = True) -> None
    def cache_key(self, request: dict, ignored_args_for_cache_key: list[str] | None = None) -> str
```

### CacheMissError

```python
class CacheMissError(Exception):
    """Raised when replay mode encounters a cache miss"""
    request: dict
    cache_key: str
    cache_file: str
```

### Environment Variables

- `CACHE_MODE`: Set to "record" or "replay" to override default mode
- Works with existing DSPy cache environment variables:
  - `DSPY_CACHEDIR`: Cache directory location
  - `DSPY_CACHE_LIMIT`: Cache size limit

## 11. Design Decisions & Rationale

### Why Context Manager?
- Natural Python idiom for resource management
- Automatically restores original cache state
- Clear scope for when replay/record is active
- Prevents accidental global cache modifications

### Why JSON over Pickle?
- Human-readable for debugging
- Version control friendly (can diff)
- No security concerns (pickle can execute code)
- Cross-platform compatible
- Easy to inspect and modify manually

### Why Cleanup Unused Entries?
- Keeps cache files minimal
- Prevents accumulation of stale data
- Makes cache files easier to review
- Reduces merge conflicts

### Why Two Modes Instead of Auto-Detection?
- Explicit is better than implicit
- Prevents accidental API calls in tests
- Clear separation of concerns
- Easier to debug when things go wrong

### Why Store Both Request and Response?
- Enables debugging (see what was cached)
- Helps diagnose cache misses
- Documents expected behavior
- Allows manual inspection/modification

## 12. Comparison with Alternatives

### vs. VCR.py
- **ReplayCache**: Designed for LM APIs, DSPy-native
- **VCR.py**: Generic HTTP recording, requires setup

### vs. pytest-recording
- **ReplayCache**: Context manager, explicit modes
- **pytest-recording**: Decorator-based, implicit

### vs. DummyLM
- **ReplayCache**: Records real LM responses
- **DummyLM**: Manual response mapping

### Advantages
- DSPy-native integration
- Simple API (single context manager)
- Automatic cleanup
- Clear error messages
- JSON format (reviewable)

## 13. Migration Guide

### From DummyLM to ReplayCache

**Before (DummyLM):**
```python
def test_qa():
    lm = DummyLM({"What is DSPy?": {"answer": "A framework"}})
    dspy.configure(lm=lm)

    program = MyProgram()
    result = program(question="What is DSPy?")
    assert result.answer == "A framework"
```

**After (ReplayCache):**
```python
def test_qa():
    # First, record with real LM
    # CACHE_MODE=record LM_FOR_TEST=openai/gpt-4o-mini pytest test_qa.py

    # Then, replay in tests
    with ReplayCache("tests/fixtures/qa.json"):
        program = MyProgram()
        result = program(question="What is DSPy?")
        assert "framework" in result.answer.lower()
```

**Benefits:**
- Real LM responses (more realistic)
- Automatic request matching
- Easy to update (just re-record)
- Shareable fixtures

---

**End of Design Document**
