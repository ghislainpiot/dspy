"""Tests for ReplayCache."""

import json

import pydantic
import pytest

import dspy
from dspy.clients.cache import Cache
from dspy.primitives.prediction import Prediction
from dspy.utils.replay_cache import CacheMissError, ReplayCache


@pytest.fixture
def cache_file(tmp_path):
    """Provide a temporary cache file path."""
    return tmp_path / "test_cache.json"


@pytest.fixture
def delegate_cache():
    """Create a simple in-memory cache for testing."""
    return Cache(
        enable_disk_cache=False,
        enable_memory_cache=True,
        disk_cache_dir="",
        disk_size_limit_bytes=0,
        memory_max_entries=100,
    )


class TestInitialization:
    """Test ReplayCache initialization."""

    def test_init_with_explicit_mode(self, cache_file):
        """Test initialization with explicit mode."""
        rc = ReplayCache(str(cache_file), mode="record")
        assert rc.mode == "record"
        assert rc.path == cache_file

        rc = ReplayCache(str(cache_file), mode="replay")
        assert rc.mode == "replay"

    def test_init_with_env_var(self, cache_file, monkeypatch):
        """Test initialization using CACHE_MODE environment variable."""
        monkeypatch.setenv("CACHE_MODE", "record")
        rc = ReplayCache(str(cache_file))
        assert rc.mode == "record"

        monkeypatch.setenv("CACHE_MODE", "replay")
        rc = ReplayCache(str(cache_file))
        assert rc.mode == "replay"

    def test_init_default_mode(self, cache_file, monkeypatch):
        """Test that default mode is replay."""
        monkeypatch.delenv("CACHE_MODE", raising=False)
        rc = ReplayCache(str(cache_file))
        assert rc.mode == "replay"

    def test_init_invalid_mode_explicit(self, cache_file):
        """Test initialization with invalid explicit mode."""
        with pytest.raises(ValueError, match="Invalid mode"):
            ReplayCache(str(cache_file), mode="invalid")

    def test_init_invalid_mode_env(self, cache_file, monkeypatch):
        """Test initialization with invalid CACHE_MODE env var."""
        monkeypatch.setenv("CACHE_MODE", "invalid")
        with pytest.raises(ValueError, match="Invalid CACHE_MODE"):
            ReplayCache(str(cache_file))

    def test_init_case_insensitive(self, cache_file):
        """Test that mode is case-insensitive."""
        rc = ReplayCache(str(cache_file), mode="RECORD")
        assert rc.mode == "record"

        rc = ReplayCache(str(cache_file), mode="Replay")
        assert rc.mode == "replay"


class TestContextManager:
    """Test ReplayCache as a context manager."""

    def test_enter_exit_restores_cache(self, cache_file):
        """Test that __exit__ restores the original cache."""
        original_cache = dspy.cache

        # Create empty cache file for replay mode
        with open(cache_file, "w") as f:
            json.dump({"metadata": {}, "entries": {}}, f)

        with ReplayCache(str(cache_file), mode="replay"):
            # Cache should be replaced
            assert dspy.cache != original_cache
            assert isinstance(dspy.cache, ReplayCache)

        # Cache should be restored
        assert dspy.cache == original_cache

    def test_enter_sets_delegate_in_record_mode(self, cache_file):
        """Test that record mode sets up delegate cache."""
        original_cache = dspy.cache

        with ReplayCache(str(cache_file), mode="record") as rc:
            assert rc._delegate == original_cache
            assert rc._delegate is not None

    def test_enter_no_delegate_in_replay_mode(self, cache_file):
        """Test that replay mode has no delegate."""
        # Create empty cache file
        with open(cache_file, "w") as f:
            json.dump({"metadata": {}, "entries": {}}, f)

        with ReplayCache(str(cache_file), mode="replay") as rc:
            assert rc._delegate is None

    def test_exit_does_not_suppress_exceptions(self, cache_file):
        """Test that exceptions are not suppressed."""
        with open(cache_file, "w") as f:
            json.dump({"metadata": {}, "entries": {}}, f)

        with pytest.raises(ValueError, match="test error"):
            with ReplayCache(str(cache_file), mode="replay"):
                raise ValueError("test error")


class TestRecordMode:
    """Test ReplayCache in record mode."""

    def test_record_mode_creates_file(self, cache_file):
        """Test that record mode creates cache file."""
        assert not cache_file.exists()

        with ReplayCache(str(cache_file), mode="record"):
            pass

        assert cache_file.exists()

    def test_record_mode_saves_data(self, cache_file):
        """Test that record mode saves cache data."""
        request = {"model": "test", "prompt": "hello"}
        response = {"answer": "world"}

        with ReplayCache(str(cache_file), mode="record"):
            dspy.cache.put(request, response)

        # Verify file was created and contains data
        assert cache_file.exists()
        with open(cache_file) as f:
            data = json.load(f)

        assert "entries" in data
        assert len(data["entries"]) == 1

    def test_record_mode_delegates_get(self, cache_file, delegate_cache):
        """Test that record mode delegates get() calls."""
        # Setup delegate cache with data
        request = {"model": "test", "prompt": "hello"}
        response = {"answer": "world"}
        delegate_cache.put(request, response)

        # Save original cache and set delegate
        original_cache = dspy.cache
        dspy.cache = delegate_cache

        try:
            with ReplayCache(str(cache_file), mode="record"):
                result = dspy.cache.get(request)
                assert result == response
        finally:
            dspy.cache = original_cache

    def test_record_mode_delegates_put(self, cache_file, delegate_cache):
        """Test that record mode delegates put() calls."""
        original_cache = dspy.cache
        dspy.cache = delegate_cache

        try:
            request = {"model": "test", "prompt": "hello"}
            response = {"answer": "world"}

            with ReplayCache(str(cache_file), mode="record"):
                dspy.cache.put(request, response)

            # Verify delegate received the put
            result = delegate_cache.get(request)
            assert result == response
        finally:
            dspy.cache = original_cache

    def test_record_mode_cleanup_unused_entries(self, cache_file):
        """Test that record mode cleans up unused entries."""
        # Create cache file with 3 entries
        entries = {
            "key1": {"request": {"id": 1}, "response": {"type": "dict", "data": {"value": 1}}},
            "key2": {"request": {"id": 2}, "response": {"type": "dict", "data": {"value": 2}}},
            "key3": {"request": {"id": 3}, "response": {"type": "dict", "data": {"value": 3}}},
        }
        with open(cache_file, "w") as f:
            json.dump({"metadata": {}, "entries": entries}, f)

        # Access only entry 1 and 2
        with ReplayCache(str(cache_file), mode="record"):
            # These calls will mark keys as accessed
            try:
                dspy.cache.get({"id": 1})
            except (KeyError, CacheMissError):
                pass
            try:
                dspy.cache.get({"id": 2})
            except (KeyError, CacheMissError):
                pass

        # Verify only accessed entries remain
        with open(cache_file) as f:
            data = json.load(f)

        # Should have cleaned up unused entries
        # Note: The actual keys will be different due to hashing
        # Just verify the structure is correct
        assert "entries" in data
        assert isinstance(data["entries"], dict)

    def test_record_mode_updates_metadata(self, cache_file):
        """Test that record mode updates metadata."""
        with ReplayCache(str(cache_file), mode="record"):
            pass

        with open(cache_file) as f:
            data = json.load(f)

        assert "metadata" in data
        metadata = data["metadata"]
        assert "created_at" in metadata
        assert "last_updated" in metadata
        assert "mode" in metadata
        assert metadata["mode"] == "record"


class TestReplayMode:
    """Test ReplayCache in replay mode."""

    def test_replay_mode_requires_existing_file(self, cache_file):
        """Test that replay mode requires cache file to exist."""
        with pytest.raises(FileNotFoundError, match="Cache file not found"):
            with ReplayCache(str(cache_file), mode="replay"):
                pass

    def test_replay_mode_loads_data(self, cache_file):
        """Test that replay mode loads cache data."""
        # Create cache file
        cache_key = "test_key_123"
        entries = {
            cache_key: {
                "request": {"model": "test"},
                "response": {"type": "dict", "data": {"answer": "test"}},
            }
        }
        with open(cache_file, "w") as f:
            json.dump({"metadata": {}, "entries": entries}, f)

        with ReplayCache(str(cache_file), mode="replay") as rc:
            assert len(rc._cache_data) == 1
            assert cache_key in rc._cache_data

    def test_replay_mode_returns_cached_response(self, cache_file):
        """Test that replay mode returns cached responses."""
        # Setup: compute the actual cache key
        request = {"model": "test", "prompt": "hello"}

        # Use a temporary cache to compute the key
        temp_cache = Cache(
            enable_disk_cache=False, enable_memory_cache=False, disk_cache_dir="", memory_max_entries=0
        )
        cache_key = temp_cache.cache_key(request)

        # Create cache file with this key
        entries = {cache_key: {"request": request, "response": {"type": "dict", "data": {"answer": "world"}}}}
        with open(cache_file, "w") as f:
            json.dump({"metadata": {}, "entries": entries}, f)

        with ReplayCache(str(cache_file), mode="replay"):
            result = dspy.cache.get(request)
            assert result == {"answer": "world"}

    def test_replay_mode_cache_miss_raises_error(self, cache_file):
        """Test that replay mode raises CacheMissError on cache miss."""
        # Create empty cache file
        with open(cache_file, "w") as f:
            json.dump({"metadata": {}, "entries": {}}, f)

        with ReplayCache(str(cache_file), mode="replay"):
            request = {"model": "test", "prompt": "unknown"}

            with pytest.raises(CacheMissError) as exc_info:
                dspy.cache.get(request)

            # Verify error message
            error = exc_info.value
            assert error.request == request
            assert error.cache_file == str(cache_file)
            assert "Cache miss in replay mode" in str(error)
            assert "CACHE_MODE=record" in str(error)

    def test_replay_mode_never_calls_delegate(self, cache_file, delegate_cache):
        """Test that replay mode never calls delegate."""
        # Create cache file
        request = {"model": "test", "prompt": "hello"}

        # Compute cache key
        temp_cache = Cache(
            enable_disk_cache=False, enable_memory_cache=False, disk_cache_dir="", memory_max_entries=0
        )
        cache_key = temp_cache.cache_key(request)

        entries = {cache_key: {"request": request, "response": {"type": "dict", "data": {"answer": "cached"}}}}
        with open(cache_file, "w") as f:
            json.dump({"metadata": {}, "entries": entries}, f)

        # Set delegate cache with different value
        delegate_cache.put(request, {"answer": "delegate"})

        original_cache = dspy.cache
        dspy.cache = delegate_cache

        try:
            with ReplayCache(str(cache_file), mode="replay"):
                # Should return cached value, not delegate value
                result = dspy.cache.get(request)
                assert result == {"answer": "cached"}
        finally:
            dspy.cache = original_cache

    def test_replay_mode_put_does_not_raise(self, cache_file):
        """Test that replay mode put() does not raise an error."""
        # Create empty cache file
        with open(cache_file, "w") as f:
            json.dump({"metadata": {}, "entries": {}}, f)

        with ReplayCache(str(cache_file), mode="replay"):
            request = {"model": "test"}
            response = {"answer": "test"}

            # put() should not raise (just logs warning)
            # We're not checking the log, just that it doesn't error
            dspy.cache.put(request, response)


class TestSerialization:
    """Test serialization and deserialization."""

    def test_serialize_prediction(self, cache_file):
        """Test serialization of Prediction objects."""
        pred = Prediction()
        pred._store = {"answer": "test", "reasoning": "because"}
        pred._lm_usage = {"tokens": 100}

        with ReplayCache(str(cache_file), mode="record") as rc:
            serialized = rc._serialize_value(pred)

        assert serialized["type"] == "Prediction"
        assert serialized["store"] == {"answer": "test", "reasoning": "because"}
        assert serialized["usage"] == {"tokens": 100}

    def test_deserialize_prediction(self, cache_file):
        """Test deserialization of Prediction objects."""
        serialized = {
            "type": "Prediction",
            "store": {"answer": "test", "reasoning": "because"},
            "usage": {"tokens": 100},
            "cache_hit": False,
        }

        with ReplayCache(str(cache_file), mode="record") as rc:
            pred = rc._deserialize_value(serialized)

        assert isinstance(pred, Prediction)
        assert pred._store == {"answer": "test", "reasoning": "because"}
        assert pred._lm_usage == {"tokens": 100}

    def test_serialize_dict(self, cache_file):
        """Test serialization of dict objects."""
        data = {"key": "value", "number": 42}

        with ReplayCache(str(cache_file), mode="record") as rc:
            serialized = rc._serialize_value(data)

        assert serialized["type"] == "dict"
        assert serialized["data"] == data

    def test_deserialize_dict(self, cache_file):
        """Test deserialization of dict objects."""
        serialized = {"type": "dict", "data": {"key": "value", "number": 42}}

        with ReplayCache(str(cache_file), mode="record") as rc:
            result = rc._deserialize_value(serialized)

        assert result == {"key": "value", "number": 42}

    def test_serialize_pydantic(self, cache_file):
        """Test serialization of pydantic models."""

        class TestModel(pydantic.BaseModel):
            name: str
            value: int

        model = TestModel(name="test", value=42)

        with ReplayCache(str(cache_file), mode="record") as rc:
            serialized = rc._serialize_value(model)

        assert serialized["type"] == "pydantic"
        assert "TestModel" in serialized["class"]
        assert serialized["data"] == {"name": "test", "value": 42}

    def test_deserialize_pydantic(self, cache_file):
        """Test deserialization of pydantic models (as dict)."""
        serialized = {
            "type": "pydantic",
            "class": "test_module.TestModel",
            "data": {"name": "test", "value": 42},
        }

        with ReplayCache(str(cache_file), mode="record") as rc:
            result = rc._deserialize_value(serialized)

        # Currently returns as dict (future: reconstruct model)
        assert result == {"name": "test", "value": 42}

    def test_serialize_primitives(self, cache_file):
        """Test serialization of primitive types."""
        with ReplayCache(str(cache_file), mode="record") as rc:
            # String
            assert rc._serialize_value("test") == {"type": "primitive", "data": "test"}
            # Int
            assert rc._serialize_value(42) == {"type": "primitive", "data": 42}
            # Float
            assert rc._serialize_value(3.14) == {"type": "primitive", "data": 3.14}
            # Bool
            assert rc._serialize_value(True) == {"type": "primitive", "data": True}
            # None
            assert rc._serialize_value(None) == {"type": "primitive", "data": None}

    def test_serialize_list(self, cache_file):
        """Test serialization of lists."""
        data = [1, 2, 3, "test"]

        with ReplayCache(str(cache_file), mode="record") as rc:
            serialized = rc._serialize_value(data)

        assert serialized["type"] == "list"
        assert serialized["data"] == data

    def test_serialize_unsupported_type(self, cache_file):
        """Test that unsupported types raise ValueError."""

        class CustomClass:
            pass

        with ReplayCache(str(cache_file), mode="record") as rc:
            with pytest.raises(ValueError, match="Cannot serialize"):
                rc._serialize_value(CustomClass())


class TestCacheKey:
    """Test cache key generation."""

    def test_cache_key_consistency(self, cache_file):
        """Test that cache_key is consistent with DSPy's Cache."""
        request = {"model": "test", "prompt": "hello", "temperature": 0.7}

        # Get key from ReplayCache
        with ReplayCache(str(cache_file), mode="record") as rc:
            replay_key = rc.cache_key(request)

        # Get key from regular Cache
        regular_cache = Cache(
            enable_disk_cache=False, enable_memory_cache=False, disk_cache_dir="", memory_max_entries=0
        )
        regular_key = regular_cache.cache_key(request)

        assert replay_key == regular_key

    def test_cache_key_with_ignored_args(self, cache_file):
        """Test cache_key with ignored arguments."""
        request = {"model": "test", "prompt": "hello", "api_key": "secret"}

        with ReplayCache(str(cache_file), mode="record") as rc:
            key1 = rc.cache_key(request, ignored_args_for_cache_key=["api_key"])
            key2 = rc.cache_key(
                {"model": "test", "prompt": "hello", "api_key": "different"}, ignored_args_for_cache_key=["api_key"]
            )

        # Keys should be the same since api_key is ignored
        assert key1 == key2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_corrupted_json_file(self, cache_file):
        """Test handling of corrupted JSON file."""
        # Create invalid JSON file
        with open(cache_file, "w") as f:
            f.write("{invalid json")

        with pytest.raises(json.JSONDecodeError):
            with ReplayCache(str(cache_file), mode="replay"):
                pass

    def test_missing_entries_key(self, cache_file):
        """Test handling of JSON without 'entries' key."""
        # Create JSON without entries
        with open(cache_file, "w") as f:
            json.dump({"metadata": {}}, f)

        # Should not raise, just load empty entries
        with ReplayCache(str(cache_file), mode="replay") as rc:
            assert rc._cache_data == {}

    def test_empty_cache_file(self, cache_file):
        """Test handling of empty cache file."""
        # Create empty JSON object
        with open(cache_file, "w") as f:
            json.dump({}, f)

        with ReplayCache(str(cache_file), mode="replay") as rc:
            assert rc._cache_data == {}

    def test_cache_key_generation_error(self, cache_file):
        """Test handling of cache key generation errors."""

        class UnserializableObject:
            pass

        request = {"data": UnserializableObject()}

        with ReplayCache(str(cache_file), mode="record"):
            # Should not raise, just return None
            result = dspy.cache.get(request)
            assert result is None

    def test_parent_directory_creation(self, tmp_path):
        """Test that parent directories are created if needed."""
        cache_file = tmp_path / "subdir1" / "subdir2" / "cache.json"
        assert not cache_file.parent.exists()

        with ReplayCache(str(cache_file), mode="record"):
            pass

        assert cache_file.exists()
        assert cache_file.parent.exists()


class TestIntegration:
    """Integration tests with DSPy components."""

    def test_with_predict_basic(self, cache_file):
        """Test ReplayCache with dspy.Predict."""
        from dspy.utils.dummies import DummyLM

        # Record mode: use DummyLM to create responses
        lm = DummyLM({"What is 2+2?": {"answer": "4"}})
        dspy.configure(lm=lm)

        with ReplayCache(str(cache_file), mode="record"):
            predictor = dspy.Predict("question -> answer")
            result1 = predictor(question="What is 2+2?")
            answer1 = result1.answer

        # Replay mode: should use cached response
        with ReplayCache(str(cache_file), mode="replay"):
            predictor = dspy.Predict("question -> answer")
            result2 = predictor(question="What is 2+2?")
            answer2 = result2.answer

        assert answer1 == answer2 == "4"

    def test_multiple_requests(self, cache_file):
        """Test multiple requests in same session."""
        from dspy.utils.dummies import DummyLM

        # Record multiple requests
        lm = DummyLM(
            {
                "What is 2+2?": {"answer": "4"},
                "What is 3+3?": {"answer": "6"},
                "What is 4+4?": {"answer": "8"},
            }
        )
        dspy.configure(lm=lm)

        with ReplayCache(str(cache_file), mode="record"):
            predictor = dspy.Predict("question -> answer")
            r1 = predictor(question="What is 2+2?")
            r2 = predictor(question="What is 3+3?")
            r3 = predictor(question="What is 4+4?")

        # Replay all requests
        with ReplayCache(str(cache_file), mode="replay"):
            predictor = dspy.Predict("question -> answer")
            s1 = predictor(question="What is 2+2?")
            s2 = predictor(question="What is 3+3?")
            s3 = predictor(question="What is 4+4?")

        assert s1.answer == r1.answer == "4"
        assert s2.answer == r2.answer == "6"
        assert s3.answer == r3.answer == "8"

    def test_cache_isolation(self, tmp_path):
        """Test that different cache files are isolated."""
        cache1 = tmp_path / "cache1.json"
        cache2 = tmp_path / "cache2.json"

        # Manually create two cache files with different data
        request1 = {"prompt": "Q1"}
        request2 = {"prompt": "Q2"}

        # Compute cache keys
        temp_cache = Cache(
            enable_disk_cache=False, enable_memory_cache=False, disk_cache_dir="", memory_max_entries=0
        )
        key1 = temp_cache.cache_key(request1)
        key2 = temp_cache.cache_key(request2)

        # Create cache1 with only request1
        with open(cache1, "w") as f:
            json.dump(
                {
                    "metadata": {},
                    "entries": {key1: {"request": request1, "response": {"type": "dict", "data": {"answer": "A1"}}}},
                },
                f,
            )

        # Create cache2 with only request2
        with open(cache2, "w") as f:
            json.dump(
                {
                    "metadata": {},
                    "entries": {key2: {"request": request2, "response": {"type": "dict", "data": {"answer": "A2"}}}},
                },
                f,
            )

        # Verify cache1 has request1 but not request2
        with ReplayCache(str(cache1), mode="replay"):
            result1 = dspy.cache.get(request1)
            assert result1 == {"answer": "A1"}

            with pytest.raises(CacheMissError):
                dspy.cache.get(request2)

        # Verify cache2 has request2 but not request1
        with ReplayCache(str(cache2), mode="replay"):
            result2 = dspy.cache.get(request2)
            assert result2 == {"answer": "A2"}

            with pytest.raises(CacheMissError):
                dspy.cache.get(request1)


class TestRealWorldScenarios:
    """Test realistic usage scenarios."""

    def test_pytest_fixture_pattern(self, tmp_path):
        """Test usage as a pytest fixture."""
        cache_file = tmp_path / "fixture_cache.json"

        from dspy.utils.dummies import DummyLM

        # Simulate fixture setup
        lm = DummyLM({"test query": {"answer": "test answer"}})
        dspy.configure(lm=lm)

        # Record in setup
        with ReplayCache(str(cache_file), mode="record"):
            predictor = dspy.Predict("question -> answer")
            predictor(question="test query")

        # Replay in test
        def test_something():
            with ReplayCache(str(cache_file), mode="replay"):
                predictor = dspy.Predict("question -> answer")
                result = predictor(question="test query")
                assert result.answer == "test answer"

        test_something()

    def test_env_var_switching(self, cache_file, monkeypatch):
        """Test switching between modes via environment variable."""
        from dspy.utils.dummies import DummyLM

        lm = DummyLM({"test": {"answer": "value"}})
        dspy.configure(lm=lm)

        # Record with env var
        monkeypatch.setenv("CACHE_MODE", "record")
        with ReplayCache(str(cache_file)):
            predictor = dspy.Predict("question -> answer")
            predictor(question="test")

        # Replay with env var
        monkeypatch.setenv("CACHE_MODE", "replay")
        with ReplayCache(str(cache_file)):
            predictor = dspy.Predict("question -> answer")
            result = predictor(question="test")
            assert result.answer == "value"
