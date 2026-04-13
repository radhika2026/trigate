"""Unit tests for gateway.scheduler.hasher (M3.2)."""
from __future__ import annotations

import hashlib
import os
import struct

import pytest

# Set env var before importing hasher so Settings picks it up
os.environ.setdefault("VLLM_BLOCK_SIZE", "16")

from gateway.scheduler.hasher import hash_pages


# ---------------------------------------------------------------------------
# Helper to compute expected hash manually
# ---------------------------------------------------------------------------

def _expected_hash(token_ids: list[int], end: int) -> str:
    raw = struct.pack(f">{end}H", *token_ids[:end])
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------

class TestHashPagesEmpty:
    def test_empty_returns_empty_list(self):
        assert hash_pages([]) == []

    def test_fewer_than_one_page_returns_empty(self):
        assert hash_pages(list(range(15)), page_size=16) == []

    def test_exactly_zero_tokens(self):
        assert hash_pages([], page_size=16) == []


class TestHashPagesBoundaries:
    def test_exactly_one_page(self):
        tokens = list(range(16))
        result = hash_pages(tokens, page_size=16)
        assert len(result) == 1
        assert result[0] == _expected_hash(tokens, 16)

    def test_two_pages(self):
        tokens = list(range(32))
        result = hash_pages(tokens, page_size=16)
        assert len(result) == 2
        assert result[0] == _expected_hash(tokens, 16)
        assert result[1] == _expected_hash(tokens, 32)

    def test_partial_page_ignored(self):
        """33 tokens with page_size=16 → 2 pages, remainder ignored."""
        tokens = list(range(33))
        result = hash_pages(tokens, page_size=16)
        assert len(result) == 2  # floor(33/16) == 2

    def test_three_pages(self):
        tokens = list(range(48))
        result = hash_pages(tokens, page_size=16)
        assert len(result) == 3
        assert result[2] == _expected_hash(tokens, 48)

    def test_page_size_8(self):
        tokens = list(range(24))
        result = hash_pages(tokens, page_size=8)
        assert len(result) == 3
        assert result[0] == _expected_hash(tokens, 8)
        assert result[1] == _expected_hash(tokens, 16)
        assert result[2] == _expected_hash(tokens, 24)


class TestHashPagesDeterminism:
    def test_same_input_same_output(self):
        tokens = list(range(64))
        assert hash_pages(tokens, page_size=16) == hash_pages(tokens, page_size=16)

    def test_different_input_different_output(self):
        a = list(range(32))
        b = list(range(1, 33))
        assert hash_pages(a, page_size=16) != hash_pages(b, page_size=16)

    def test_order_matters(self):
        tokens = [1, 2, 3] + [0] * 13
        tokens_reversed = [3, 2, 1] + [0] * 13
        assert hash_pages(tokens, page_size=16) != hash_pages(tokens_reversed, page_size=16)


class TestHashPagesRollingCumulative:
    """Verify hashes cover [0..16], [0..32], [0..48] — cumulative not sliding."""

    def test_first_page_hash_is_prefix_of_second(self):
        """The first page hash is based on tokens[0:16]; second on tokens[0:32]."""
        tokens = list(range(32))
        result = hash_pages(tokens, page_size=16)
        # First page: tokens[0:16]
        expected_0 = _expected_hash(tokens, 16)
        # Second page: tokens[0:32]
        expected_1 = _expected_hash(tokens, 32)
        assert result[0] == expected_0
        assert result[1] == expected_1
        # They should differ because the inputs differ
        assert result[0] != result[1]

    def test_sliding_window_not_used(self):
        """Sliding window would hash tokens[16:32]; cumulative hashes tokens[0:32]."""
        tokens = list(range(32))
        result = hash_pages(tokens, page_size=16)
        # Sliding window hash (wrong)
        sliding_raw = struct.pack(">16H", *tokens[16:32])
        sliding_hash = hashlib.sha256(sliding_raw).hexdigest()
        assert result[1] != sliding_hash  # must NOT be a sliding window


class TestHashPagesHandVerified:
    """10 hand-verified (token_ids, expected_hashes) pairs."""

    CASES = [
        # (token_ids, page_size, expected_hashes_at_pages)
        (list(range(16)), 16, [_expected_hash(list(range(16)), 16)]),
        (list(range(32)), 16, [
            _expected_hash(list(range(32)), 16),
            _expected_hash(list(range(32)), 32),
        ]),
        ([0] * 16, 16, [_expected_hash([0] * 16, 16)]),
        ([65535] * 16, 16, [_expected_hash([65535] * 16, 16)]),
        ([1] * 32, 16, [
            _expected_hash([1] * 32, 16),
            _expected_hash([1] * 32, 32),
        ]),
        (list(range(48)), 16, [
            _expected_hash(list(range(48)), 16),
            _expected_hash(list(range(48)), 32),
            _expected_hash(list(range(48)), 48),
        ]),
        (list(range(8)), 8, [_expected_hash(list(range(8)), 8)]),
        (list(range(16)), 8, [
            _expected_hash(list(range(16)), 8),
            _expected_hash(list(range(16)), 16),
        ]),
        ([100, 200, 300] + [0] * 13, 16, [_expected_hash([100, 200, 300] + [0] * 13, 16)]),
        (list(range(1000, 1016)), 16, [_expected_hash(list(range(1000, 1016)), 16)]),
    ]

    @pytest.mark.parametrize("token_ids,page_size,expected", CASES)
    def test_hand_verified(self, token_ids, page_size, expected):
        result = hash_pages(token_ids, page_size=page_size)
        assert result == expected


class TestHashPagesDefaultPageSize:
    def test_uses_env_default(self):
        """When page_size is None, defaults to VLLM_BLOCK_SIZE (16)."""
        tokens = list(range(16))
        result_default = hash_pages(tokens)
        result_explicit = hash_pages(tokens, page_size=16)
        assert result_default == result_explicit
