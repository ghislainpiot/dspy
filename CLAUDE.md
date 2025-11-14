# CLAUDE.md - DSPy Codebase Guide for AI Assistants

This document provides a comprehensive guide to the DSPy codebase for AI assistants working on this repository. It covers architecture, conventions, workflows, and best practices.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Core Concepts & Abstractions](#core-concepts--abstractions)
4. [Development Workflow](#development-workflow)
5. [Code Style & Conventions](#code-style--conventions)
6. [Testing Guidelines](#testing-guidelines)
7. [Common Patterns](#common-patterns)
8. [Where to Find Things](#where-to-find-things)
9. [Making Changes - Common Scenarios](#making-changes---common-scenarios)
10. [Tips for AI Assistants](#tips-for-ai-assistants)

---

## Project Overview

**DSPy** (Declarative Self-improving Python) is a framework for programming—rather than prompting—language models. It allows developers to build modular AI systems and provides algorithms for optimizing prompts and weights.

- **Documentation:** https://dspy.ai
- **Version:** 3.0.4b2
- **Python:** 3.10-3.14
- **License:** MIT-style (see LICENSE)

### Key Features
- Compositional Python code instead of brittle prompts
- Signature-based abstraction for LM inputs/outputs
- Automatic prompt optimization (teleprompters)
- Support for multiple LM providers via LiteLLM
- Built-in evaluation and metrics
- Modular architecture with reusable components

---

## Repository Structure

```
dspy/
├── dspy/                      # Main package
│   ├── __init__.py           # Top-level exports
│   ├── primitives/           # Core abstractions (Module, Example, Prediction)
│   ├── signatures/           # Signature system (Pydantic-based)
│   ├── predict/              # Predictor modules (Predict, CoT, ReAct, etc.)
│   ├── clients/              # LM client abstraction (LiteLLM-based)
│   ├── adapters/             # Format adapters (Chat, JSON, XML, etc.)
│   ├── teleprompt/           # Optimizers (Bootstrap, MIPRO, COPRO, GEPA, etc.)
│   ├── evaluate/             # Evaluation framework
│   ├── retrievers/           # Retrieval modules
│   ├── datasets/             # Dataset loaders
│   ├── utils/                # Utilities (callbacks, caching, async, etc.)
│   ├── streaming/            # Streaming support
│   ├── propose/              # Proposal generation
│   ├── experimental/         # Experimental features
│   └── dsp/                  # Legacy compatibility layer
├── tests/                    # Test suite (mirrors dspy/ structure)
├── docs/                     # Documentation (MkDocs)
├── pyproject.toml            # Project configuration
├── CONTRIBUTING.md           # Contribution guidelines
├── .pre-commit-config.yaml   # Pre-commit hooks (ruff)
└── uv.lock                   # UV lock file
```

### Key Directories

#### `dspy/primitives/`
Core building blocks:
- `module.py` - Base `Module` class with metaclass `ProgramMeta`
- `base_module.py` - Base functionality (save/load, parameters, deepcopy)
- `example.py` - `Example` class (flexible data container)
- `prediction.py` - `Prediction` class (output container)
- `python_interpreter.py` - Python code execution

#### `dspy/signatures/`
Signature system for defining LM I/O:
- `signature.py` - `Signature` base class (Pydantic-based)
- `field.py` - `InputField`, `OutputField` definitions

#### `dspy/predict/`
Predictor modules:
- `predict.py` - Base `Predict` class
- `chain_of_thought.py` - Chain-of-Thought reasoning
- `react.py` - ReAct agent pattern
- `program_of_thought.py` - Program-of-Thought
- `retry.py`, `refine.py`, `parallel.py` - Execution patterns
- `aggregation.py`, `best_of_n.py` - Output selection
- `avatar/` - Avatar optimizer module

#### `dspy/clients/`
LM client abstraction:
- `lm.py` - Main `LM` class using LiteLLM
- `base_lm.py` - Abstract base class
- `provider.py` - Provider interface
- `cache.py` - Disk and memory caching
- `openai.py`, `databricks.py` - Specific providers

#### `dspy/adapters/`
Transform between DSPy and LM formats:
- `base.py` - Base `Adapter` class
- `chat_adapter.py` - Default chat-based adapter
- `json_adapter.py`, `xml_adapter.py` - Alternative formats
- `types/` - Custom type system (Image, Audio, Tool, Reasoning, etc.)

#### `dspy/teleprompt/`
Optimization algorithms (teleprompters):
- `bootstrap.py` - `BootstrapFewShot` (core bootstrapping)
- `mipro_optimizer_v2.py` - MIPROv2
- `copro_optimizer.py` - COPRO
- `gepa/` - GEPA (Reflective Prompt Evolution)
- `signature_opt.py` - Signature optimization
- `knn_fewshot.py`, `vanilla.py` - Simple optimizers
- `ensemble.py` - Ensemble methods

#### `dspy/evaluate/`
Evaluation framework:
- `evaluate.py` - `Evaluate` class with parallel execution
- `metrics.py` - Common metrics
- `auto_evaluation.py` - Automated evaluation

---

## Core Concepts & Abstractions

### 1. Module Hierarchy

```python
BaseModule (base_module.py)
└── Module (module.py)
    ├── Predict (predict/predict.py) + Parameter
    ├── ChainOfThought
    ├── ReAct
    └── [Other predictors]
```

**Key characteristics:**
- All modules inherit from `Module` (which inherits from `BaseModule`)
- Modules implement `forward()` and optionally `aforward()` (async)
- Call modules via `module()`, not `module.forward()` (warning emitted)
- Modules can contain sub-modules (compositional)
- Support for callbacks, history tracking, and compilation state

### 2. Signature System

Signatures define the inputs and outputs for language model calls using Pydantic v2:

```python
# Class-based signature
class QA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

# String-based signature (equivalent)
dspy.Signature("question -> answer")
```

**Key features:**
- Built on Pydantic `BaseModel` with `SignatureMeta` metaclass
- Fields marked with `InputField()` or `OutputField()`
- Automatic prefix inference (e.g., `camelCase` → "Camel Case:")
- Instructions in `__doc__`
- Methods: `with_instructions()`, `with_updated_fields()`, `prepend()`, `append()`
- Custom types: `Image`, `Audio`, `Tool`, `Reasoning`, etc.

### 3. Predict & Parameter

`Predict` is the fundamental predictor class:
- Inherits from both `Module` and `Parameter`
- `Parameter` marks optimizable components
- Handles LM configuration, signature management, demos
- `forward()` and `aforward()` methods
- State serialization via `dump_state()`/`load_state()`

### 4. Example & Prediction

**Example**: Flexible data container
- Dictionary-like with attribute access
- `with_inputs()` - mark fields as inputs
- Methods: `inputs()`, `labels()`, `without()`, `toDict()`

**Prediction**: Output container (extends Example)
- Supports multiple completions via `Completions`
- Comparison/arithmetic on score fields
- LM usage tracking

### 5. Adapters

Adapters transform between DSPy and LM-specific formats:
- `ChatAdapter` - Default for chat models
- `JSONAdapter` - JSON I/O
- `XMLAdapter` - XML I/O
- `TwoStepAdapter` - Two-stage processing
- Custom type handling (Image, Audio, Tool, etc.)
- Native function calling support

### 6. Teleprompters (Optimizers)

Teleprompters optimize DSPy programs:
- Base class: `Teleprompter` with `compile(student, trainset, teacher, valset)` method
- **BootstrapFewShot**: Generate high-quality demos by running teacher
- **MIPROv2**: Multi-stage instruction/demo optimization
- **COPRO**: LM-generated instruction improvements
- **GEPA**: Reflective prompt evolution
- Teacher-student paradigm with metric-based validation

### 7. Configuration System

Global configuration via `dspy.settings`:
- `dspy.configure(lm=..., rm=...)` - Set global config
- `dspy.context(lm=..., adapter=...)` - Thread-local override (context manager)
- Thread-safe with `contextvars`
- Only owner thread can call `configure()`

---

## Development Workflow

### 1. Environment Setup

**Requirements:** Python 3.10+

**Option A: Using uv (recommended)**
```bash
uv venv --python 3.10
uv sync --extra dev
uv run pytest tests/predict
```

**Option B: Using conda + pip**
```bash
conda create -n dspy-dev python=3.10
conda activate dspy-dev
pip install -e ".[dev]"
pytest tests/predict
```

### 2. Pre-commit Hooks

**Setup (required):**
```bash
pre-commit install
```

**Auto-runs on commit:**
- `ruff check --fix-only` - Linting with auto-fix
- `check-yaml`, `check-toml` - Config validation
- `check-added-large-files` - Prevent large files
- `check-merge-conflict` - Detect conflicts
- `debug-statements` - Find leftover debug code

**Manual runs:**
```bash
pre-commit run                    # Check staged files
pre-commit run --all-files        # Check all files
pre-commit run --files path/to/file.py
```

### 3. Making Changes

1. **Open an issue** (for non-trivial changes)
2. **Fork and clone** the repository
3. **Create a branch** for your feature/fix
4. **Make changes** following code conventions
5. **Run tests** - ensure all pass
6. **Commit with pre-commit** - hooks will auto-format
7. **Create PR** to `main` branch
8. **Code review** - iterate on feedback
9. **Merge** - maintainer will merge when approved

### 4. Running Tests

```bash
# Run all tests (fast, uses DummyLM)
pytest tests/

# Run specific test file
pytest tests/predict/test_predict.py

# Run with markers (requires explicit flags)
pytest --reliability tests/       # Reliability tests
pytest --extra tests/              # Extra tests
pytest --llm_call tests/           # Tests requiring real LM calls

# Run with coverage
pytest --cov=dspy tests/

# Set LM for integration tests
export LM_FOR_TEST="openai/gpt-4o-mini"
pytest --llm_call tests/
```

---

## Code Style & Conventions

### 1. Style Guide

Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) with **ruff** enforcement.

**Configuration:** `pyproject.toml` → `[tool.ruff]`
- Line length: 120 characters
- Python 3.10+ features allowed
- Linters: pycodestyle, pyflakes, isort, flake8-comprehensions, pyupgrade, ruff-specific

### 2. Naming Conventions

- **Classes:** `PascalCase` (e.g., `Predict`, `ChainOfThought`)
- **Functions/Methods:** `snake_case` (e.g., `forward`, `dump_state`)
- **Files:** `snake_case` (e.g., `chain_of_thought.py`)
- **Constants:** `UPPER_SNAKE_CASE` (e.g., `DEFAULT_CONFIG`)
- **Private/Internal:** Leading underscore (e.g., `_base_init`, `_compiled`)

### 3. Import Conventions

```python
# Prefer absolute imports
from dspy.primitives.module import Module  # Good
from .module import Module                 # Avoid (except in __init__.py)

# Top-level __init__.py uses wildcard imports
from dspy.predict import *
from dspy.primitives import *

# Order: stdlib, third-party, dspy
import os
import pytest
from dspy.primitives import Module
```

### 4. Docstrings

Use Google-style docstrings:
```python
def forward(self, **kwargs):
    """Execute the predictor with given inputs.

    Args:
        **kwargs: Input fields matching the signature.

    Returns:
        Prediction: Output with fields from signature.

    Raises:
        ValueError: If required inputs are missing.
    """
```

### 5. Type Hints

Use type hints where helpful:
```python
def forward(self, **kwargs) -> Prediction:
    ...

# For complex types
from typing import Optional, Dict, Any

def configure(lm: Optional[LM] = None) -> Dict[str, Any]:
    ...
```

---

## Testing Guidelines

### 1. Test Structure

Tests mirror the `dspy/` structure:
```
tests/
├── primitives/       # Module, Example, Prediction
├── predict/          # Predictor tests
├── signatures/       # Signature tests
├── teleprompt/       # Optimizer tests
├── clients/          # LM client tests
├── adapters/         # Adapter tests
├── utils/            # Utility tests
├── evaluate/         # Evaluation tests
└── conftest.py       # Pytest configuration
```

### 2. Test Fixtures

**Available fixtures (from `conftest.py`):**
- `clear_settings` - Auto-used, resets config after each test
- `lm_for_test` - Gets LM from `LM_FOR_TEST` env var
- `litellm_test_server` - Test server for integration tests
- `anyio_backend` - "asyncio" for async tests

### 3. Test Markers

**Default skip markers** (require explicit flags):
- `@pytest.mark.reliability` - Run with `--reliability`
- `@pytest.mark.extra` - Run with `--extra`
- `@pytest.mark.llm_call` - Run with `--llm_call`

Example:
```python
@pytest.mark.llm_call
def test_with_real_lm(lm_for_test):
    dspy.configure(lm=dspy.LM(lm_for_test))
    # Test with real LM
```

### 4. DummyLM Pattern

Use `DummyLM` for deterministic tests:
```python
from dspy.utils.dummies import DummyLM

def test_forward():
    lm = DummyLM({"What is 1+1?": {"answer": "2"}})
    dspy.configure(lm=lm)

    program = SimpleQA()
    result = program(question="What is 1+1?")

    assert result.answer == "2"
```

### 5. Async Testing

Use `pytest-asyncio`:
```python
import pytest

@pytest.mark.asyncio
async def test_async_forward():
    program = AsyncProgram()
    result = await program.aforward(question="test")
    assert result.answer
```

### 6. Test Conventions

- Test files: `test_*.py`
- Test functions: `test_*`
- Use descriptive names: `test_forward_with_multiple_outputs`
- One concept per test
- Use fixtures for setup
- Mock external dependencies (LMs, APIs)
- Test edge cases and error conditions

---

## Common Patterns

### 1. Module Pattern

All modules follow this pattern:
```python
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predictor(question=question)

# Usage
module = MyModule()
result = module(question="What is DSPy?")  # Don't call .forward() directly
```

### 2. Signature Definition

```python
# Option 1: String-based (simple)
sig = dspy.Signature("question, context -> answer")

# Option 2: Class-based (with types and descriptions)
class QASignature(dspy.Signature):
    """Answer questions based on context."""

    question: str = dspy.InputField(desc="The question to answer")
    context: str = dspy.InputField(desc="Relevant context")
    answer: str = dspy.OutputField(desc="A concise answer")

# Option 3: Dynamic modification
sig = QASignature.with_instructions("Be concise and factual")
sig = sig.with_updated_fields("answer", desc="A detailed answer")
```

### 3. Configuration Pattern

```python
# Global configuration
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Thread-local override
with dspy.context(lm=dspy.LM("openai/gpt-4o")):
    result = expensive_program(query="...")

# Outside context, uses global config again
```

### 4. Optimization Pattern

```python
# Define program
class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Optimize
optimizer = dspy.BootstrapFewShot(metric=exact_match)
optimized_rag = optimizer.compile(RAG(), trainset=train)

# Save
optimized_rag.save("rag_optimized.json")
```

### 5. State Management Pattern

```python
# Save state (JSON preferred)
module.save("model.json")                           # State only
module.save("model/", save_program=True)            # State + architecture

# Load state
module = dspy.load("model.json")                    # If architecture unchanged
module = MyModule()
module.load("model.json")                           # Load into existing instance
```

### 6. Callback Pattern

```python
from dspy.utils.callback import with_callbacks

@with_callbacks
def custom_function(x):
    # Callbacks automatically triggered
    return x * 2

# Or for classes
class CustomPredictor(dspy.Module):
    @with_callbacks
    def forward(self, **kwargs):
        # Callbacks automatically applied
        return self.predictor(**kwargs)
```

### 7. Async Pattern

```python
class AsyncModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("question -> answer")

    async def aforward(self, question):
        # Use aforward for async execution
        return await self.predictor.aforward(question=question)

# Usage
result = await async_module(question="...")  # __call__ detects async context
```

### 8. History Inspection Pattern

```python
program = MyProgram()
result = program(question="What is DSPy?")

# Inspect last LM call
program.inspect_history(n=1)

# Access full history
for call in program._module_history:
    print(call.inputs, call.outputs)
```

---

## Where to Find Things

### Looking for...

**Core abstractions:**
- Module base class → `dspy/primitives/module.py`
- Signature system → `dspy/signatures/signature.py`
- Example/Prediction → `dspy/primitives/example.py`, `prediction.py`

**Predictor modules:**
- Base Predict → `dspy/predict/predict.py`
- Chain-of-Thought → `dspy/predict/chain_of_thought.py`
- ReAct agent → `dspy/predict/react.py`
- All predictors → `dspy/predict/`

**LM client:**
- Main LM class → `dspy/clients/lm.py`
- Provider interface → `dspy/clients/provider.py`
- Caching → `dspy/clients/cache.py`

**Optimizers:**
- Bootstrap → `dspy/teleprompt/bootstrap.py`
- MIPRO → `dspy/teleprompt/mipro_optimizer_v2.py`
- All optimizers → `dspy/teleprompt/`

**Evaluation:**
- Evaluate class → `dspy/evaluate/evaluate.py`
- Metrics → `dspy/evaluate/metrics.py`

**Utilities:**
- Callbacks → `dspy/utils/callback.py`
- Async/sync conversion → `dspy/utils/asyncify.py`, `syncify.py`
- Saving/loading → `dspy/utils/saving.py`
- Caching → `dspy/utils/caching.py`

**Configuration:**
- Settings → `dspy/dsp/utils/settings.py`
- configure/context → Aliased in `dspy/__init__.py`

**Custom types:**
- Image, Audio, Tool, etc. → `dspy/adapters/types/`

**Tests:**
- Mirror structure of `dspy/`
- Fixtures → `tests/conftest.py`
- Test server → `tests/test_utils/server.py`

---

## Making Changes - Common Scenarios

### Scenario 1: Adding a New Predictor

**Location:** `dspy/predict/my_predictor.py`

```python
from dspy.primitives.module import Module
from dspy.predict.predict import Predict

class MyPredictor(Module):
    def __init__(self, signature, **config):
        super().__init__()
        # Modify signature if needed
        self.signature = signature
        self.predict = Predict(signature, **config)

    def forward(self, **kwargs):
        # Custom logic
        return self.predict(**kwargs)
```

**Don't forget:**
1. Add to `dspy/predict/__init__.py`
2. Add tests in `tests/predict/test_my_predictor.py`
3. Update documentation in `docs/`

### Scenario 2: Adding a New Optimizer

**Location:** `dspy/teleprompt/my_optimizer.py`

```python
from dspy.teleprompt.teleprompt import Teleprompter

class MyOptimizer(Teleprompter):
    def __init__(self, metric, **kwargs):
        self.metric = metric
        self.kwargs = kwargs

    def compile(self, student, trainset, teacher=None, valset=None):
        # Optimization logic
        # Return optimized student
        return student
```

**Don't forget:**
1. Add to `dspy/teleprompt/__init__.py`
2. Add tests in `tests/teleprompt/test_my_optimizer.py`
3. Add tutorial/example in `docs/tutorials/`

### Scenario 3: Adding Support for a New LM Provider

**Location:** `dspy/clients/my_provider.py`

```python
from dspy.clients.provider import Provider

class MyProvider(Provider):
    def __call__(self, prompt, **kwargs):
        # Provider-specific API call
        return response
```

**Integration:**
1. Add to `dspy/clients/__init__.py`
2. Update `dspy/clients/lm.py` if needed
3. Add tests in `tests/clients/test_my_provider.py`
4. Document in `docs/api/clients/`

### Scenario 4: Adding a Custom Type

**Location:** `dspy/adapters/types/my_type.py`

```python
from pydantic import BaseModel

class MyType(BaseModel):
    """Custom type for special data."""
    data: str
    metadata: dict
```

**Don't forget:**
1. Add to `dspy/adapters/__init__.py`
2. Update adapter formatting in `dspy/adapters/chat_adapter.py`
3. Add tests in `tests/adapters/types/test_my_type.py`

### Scenario 5: Fixing a Bug

**Process:**
1. **Identify location** - Use grep/search to find relevant code
2. **Write a failing test** - Reproduce the bug in `tests/`
3. **Fix the bug** - Make minimal changes to fix
4. **Verify test passes** - Run `pytest tests/path/to/test.py`
5. **Check for regressions** - Run full test suite
6. **Submit PR** - Reference issue number

### Scenario 6: Adding a New Metric

**Location:** `dspy/evaluate/metrics.py`

```python
def my_metric(example, pred, trace=None):
    """Compute custom metric.

    Args:
        example: Gold example with labels
        pred: Prediction from model
        trace: Optional execution trace

    Returns:
        float: Score between 0 and 1
    """
    # Metric logic
    return score
```

**Usage:**
```python
evaluator = dspy.Evaluate(devset=devset, metric=my_metric)
score = evaluator(program)
```

---

## Tips for AI Assistants

### 1. Understanding the Codebase

- **Start with:** `dspy/__init__.py` to see top-level exports
- **Core classes:** Module, Signature, Predict, Example, Prediction
- **Read tests:** Tests show usage patterns and edge cases
- **Check types:** Most code uses type hints and Pydantic

### 2. Making Changes

- **Read before writing:** Always read existing code first
- **Follow patterns:** Match existing code style and structure
- **Test-driven:** Write tests alongside code changes
- **Small PRs:** Keep changes focused and reviewable
- **Documentation:** Update docs when adding features

### 3. Common Pitfalls

- **Don't bypass forward():** Always call `module()`, not `module.forward()`
- **Don't modify _compiled:** Compilation state is managed internally
- **Don't skip pre-commit:** Hooks enforce consistency
- **Don't use global state carelessly:** Use `dspy.context()` for local overrides
- **Don't forget async:** If adding async support, implement both `forward()` and `aforward()`

### 4. Code Search Strategies

```bash
# Find class definitions
grep -r "class MyClass" dspy/

# Find usage of a function
grep -r "my_function(" dspy/ tests/

# Find imports
grep -r "from dspy.predict import" dspy/

# Find similar test patterns
grep -r "def test_forward" tests/
```

### 5. Understanding Relationships

**Module → Predict → LM flow:**
```
MyModule.forward()
  → Predict.forward()
    → Adapter.format()
      → LM.__call__()
        → Provider API
      → Response
    → Adapter.parse()
  → Prediction
```

**Optimization flow:**
```
Teleprompter.compile(student, trainset)
  → Teacher.forward() on trainset
  → Validate with metric
  → Update student demos/prompts
  → Return optimized student
```

### 6. Debugging Tips

- **Enable logging:** `dspy.enable_logging()`
- **Inspect history:** `module.inspect_history(n=1)`
- **Use DummyLM:** Deterministic responses for testing
- **Check settings:** `dspy.settings` holds global config
- **Print signatures:** `print(module.signature.fields)`

### 7. Performance Considerations

- **Caching:** Enabled by default in `~/.dspy_cache/`
- **Parallel execution:** Use `Evaluate(num_threads=N)`
- **Async support:** Prefer `aforward()` for concurrent calls
- **Batch processing:** Use `module.batch()` for multiple inputs

### 8. Architecture Principles

- **Compositionality:** Modules contain modules
- **Declarative:** Define what, not how (LM figures out how)
- **Optimization:** Programs can be automatically improved
- **Modularity:** Swap components (LMs, adapters, optimizers)
- **Type safety:** Pydantic validation on inputs/outputs

### 9. When to Ask for Clarification

- **Ambiguous requirements:** "Should this use CoT or basic Predict?"
- **Breaking changes:** "This will change the API, is that OK?"
- **Performance tradeoffs:** "Caching vs. freshness?"
- **Design decisions:** "New module or extend existing?"
- **Scope questions:** "Should this also handle edge case X?"

### 10. Resources

- **Documentation:** https://dspy.ai (comprehensive guides)
- **Discord:** Active community for questions
- **Issues:** Check existing issues before creating
- **Papers:** See README for research papers on DSPy
- **Examples:** `docs/tutorials/` has 45+ tutorials

---

## Quick Reference

### Common Commands

```bash
# Setup
uv venv --python 3.10
uv sync --extra dev
pre-commit install

# Testing
pytest tests/
pytest tests/predict/test_predict.py
pytest --llm_call tests/

# Linting
pre-commit run --all-files
ruff check dspy/ tests/
ruff format dspy/ tests/

# Building docs
cd docs
mkdocs serve
```

### Common Imports

```python
import dspy
from dspy import Predict, ChainOfThought, ReAct
from dspy import Signature, InputField, OutputField
from dspy import Example, Prediction
from dspy import Module
from dspy import BootstrapFewShot, MIPROv2
from dspy import Evaluate
from dspy.utils.dummies import DummyLM
```

### File Locations (Quick)

- Main package: `dspy/`
- Tests: `tests/`
- Docs: `docs/`
- Config: `pyproject.toml`
- Pre-commit: `.pre-commit-config.yaml`
- Contributing: `CONTRIBUTING.md`

---

## Version History

- **v3.0.4b2** (current) - Latest beta release
- Major changes in 3.x:
  - Pydantic v2 migration
  - New adapter system
  - Enhanced async support
  - Improved caching
  - MCP support

---

## Contact & Support

- **Issues:** https://github.com/stanfordnlp/dspy/issues
- **Discord:** https://discord.gg/XCGy2WDCQB
- **Twitter:** @DSPyOSS
- **Docs:** https://dspy.ai

---

**Last Updated:** 2025-11-14
**For:** DSPy v3.0.4b2
**Python:** 3.10-3.14
