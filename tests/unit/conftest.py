"""Unit test configuration.

Pre-imports torch/sentence_transformers BEFORE any threads are spawned
to avoid MPS abort on macOS when torch is first imported after threads exist.
"""
from __future__ import annotations

import os

# Must be set before ANY torch import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

# Force CPU — avoids MPS initialization errors in test environment
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Pre-import torch eagerly so it runs before any ThreadPoolExecutor is created.
# This prevents the "Aborted" crash when torch/MPS is first loaded after threads.
try:
    import torch  # noqa: F401
except Exception:
    pass
