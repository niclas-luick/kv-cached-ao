"""Tests for vectorized create_kv_attention_mask and KVCacheOracle boundary detection."""

import pytest
import torch

from nl_probes.utils.dataset_utils import KVCacheBatchData, create_kv_attention_mask
from experiments.kv_cache_inference import KVCacheOracle


def _reference_create_kv_attention_mask(
    batch: KVCacheBatchData,
    attend_full_context: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Original loop-based implementation, copied verbatim for reference testing."""
    B = batch.oracle_input_ids.shape[0]
    O = batch.oracle_input_ids.shape[1]
    C = batch.context_input_ids.shape[1]

    if device is None:
        device = batch.oracle_input_ids.device

    mask = torch.full((B, 1, O, C + O), float("-inf"), device=device, dtype=dtype)

    for b in range(B):
        ctx_pad_len = batch.context_padding_lengths[b]
        oracle_pad_len = (batch.oracle_attention_mask[b] == False).sum().item()

        if attend_full_context:
            ctx_start = ctx_pad_len
            ctx_end = C
            mask[b, 0, :, ctx_start:ctx_end] = 0.0
        else:
            for pos in batch.context_positions[b]:
                adjusted_pos = pos + ctx_pad_len
                if 0 <= adjusted_pos < C:
                    mask[b, 0, :, adjusted_pos] = 0.0

        for i in range(O):
            if i >= oracle_pad_len:
                for j in range(oracle_pad_len, i + 1):
                    mask[b, 0, i, C + j] = 0.0

    return mask


def _make_batch(
    context_lengths: list[int],
    oracle_lengths: list[int],
    context_positions_list: list[list[int]],
    device: torch.device = torch.device("cpu"),
) -> KVCacheBatchData:
    """Helper to build a KVCacheBatchData with given lengths and positions."""
    B = len(context_lengths)
    max_ctx = max(context_lengths)
    max_oracle = max(oracle_lengths)

    ctx_ids = []
    ctx_masks = []
    oracle_ids = []
    oracle_labels = []
    oracle_masks = []
    ctx_pad_lengths = []

    for b in range(B):
        ctx_len = context_lengths[b]
        ctx_pad = max_ctx - ctx_len
        ctx_ids.append(torch.cat([torch.zeros(ctx_pad, dtype=torch.long), torch.arange(1, ctx_len + 1)]).to(device))
        ctx_masks.append(torch.cat([torch.zeros(ctx_pad, dtype=torch.bool), torch.ones(ctx_len, dtype=torch.bool)]).to(device))
        ctx_pad_lengths.append(ctx_pad)

        o_len = oracle_lengths[b]
        o_pad = max_oracle - o_len
        oracle_ids.append(torch.cat([torch.zeros(o_pad, dtype=torch.long), torch.arange(1, o_len + 1)]).to(device))
        oracle_labels.append(torch.cat([torch.full((o_pad,), -100, dtype=torch.long), torch.arange(1, o_len + 1)]).to(device))
        oracle_masks.append(torch.cat([torch.zeros(o_pad, dtype=torch.bool), torch.ones(o_len, dtype=torch.bool)]).to(device))

    return KVCacheBatchData(
        context_input_ids=torch.stack(ctx_ids),
        context_attention_mask=torch.stack(ctx_masks),
        oracle_input_ids=torch.stack(oracle_ids),
        oracle_labels=torch.stack(oracle_labels),
        oracle_attention_mask=torch.stack(oracle_masks),
        context_positions=context_positions_list,
        context_padding_lengths=ctx_pad_lengths,
    )


class TestVectorizedMaskMatchesReference:
    """Verify vectorized create_kv_attention_mask matches the original loop-based version."""

    def test_attend_full_context_uniform_lengths(self):
        batch = _make_batch(
            context_lengths=[10, 10],
            oracle_lengths=[5, 5],
            context_positions_list=[[0, 1, 2], [7, 8, 9]],
        )
        ref = _reference_create_kv_attention_mask(batch, attend_full_context=True)
        vec = create_kv_attention_mask(batch, attend_full_context=True)
        assert torch.allclose(ref, vec, atol=1e-6), f"Max diff: {(ref - vec).abs().max()}"

    def test_attend_full_context_varied_lengths(self):
        batch = _make_batch(
            context_lengths=[15, 8, 20],
            oracle_lengths=[10, 6, 3],
            context_positions_list=[[0, 1], [3, 4, 5], [10]],
        )
        ref = _reference_create_kv_attention_mask(batch, attend_full_context=True)
        vec = create_kv_attention_mask(batch, attend_full_context=True)
        assert torch.allclose(ref, vec, atol=1e-6), f"Max diff: {(ref - vec).abs().max()}"

    def test_selective_positions_uniform(self):
        batch = _make_batch(
            context_lengths=[10, 10],
            oracle_lengths=[5, 5],
            context_positions_list=[[3, 4, 5], [7, 8]],
        )
        ref = _reference_create_kv_attention_mask(batch, attend_full_context=False)
        vec = create_kv_attention_mask(batch, attend_full_context=False)
        assert torch.allclose(ref, vec, atol=1e-6), f"Max diff: {(ref - vec).abs().max()}"

    def test_selective_positions_varied(self):
        batch = _make_batch(
            context_lengths=[12, 7, 20, 5],
            oracle_lengths=[8, 3, 15, 10],
            context_positions_list=[[0], [2, 3, 4], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4]],
        )
        ref = _reference_create_kv_attention_mask(batch, attend_full_context=False)
        vec = create_kv_attention_mask(batch, attend_full_context=False)
        assert torch.allclose(ref, vec, atol=1e-6), f"Max diff: {(ref - vec).abs().max()}"

    def test_single_item_batch(self):
        batch = _make_batch(
            context_lengths=[30],
            oracle_lengths=[20],
            context_positions_list=[[10, 11, 12, 13, 14]],
        )
        for attend_full in [True, False]:
            ref = _reference_create_kv_attention_mask(batch, attend_full_context=attend_full)
            vec = create_kv_attention_mask(batch, attend_full_context=attend_full)
            assert torch.allclose(ref, vec, atol=1e-6), f"attend_full={attend_full}, max diff: {(ref - vec).abs().max()}"

    def test_single_token_context_position(self):
        batch = _make_batch(
            context_lengths=[5, 8],
            oracle_lengths=[3, 4],
            context_positions_list=[[2], [5]],
        )
        ref = _reference_create_kv_attention_mask(batch, attend_full_context=False)
        vec = create_kv_attention_mask(batch, attend_full_context=False)
        assert torch.allclose(ref, vec, atol=1e-6), f"Max diff: {(ref - vec).abs().max()}"

    def test_large_batch(self):
        B = 16
        ctx_lens = [10 + i * 3 for i in range(B)]
        oracle_lens = [5 + i * 2 for i in range(B)]
        positions = [list(range(i, min(i + 5, ctx_lens[i]))) for i in range(B)]
        batch = _make_batch(ctx_lens, oracle_lens, positions)

        for attend_full in [True, False]:
            ref = _reference_create_kv_attention_mask(batch, attend_full_context=attend_full)
            vec = create_kv_attention_mask(batch, attend_full_context=attend_full)
            assert torch.allclose(ref, vec, atol=1e-6), f"attend_full={attend_full}, max diff: {(ref - vec).abs().max()}"


class TestMaskShapeAndValues:
    """Verify mask shape and that values are only 0.0 or -inf."""

    def test_shape(self):
        batch = _make_batch([10, 8], [5, 7], [[0, 1], [3, 4, 5]])
        mask = create_kv_attention_mask(batch)
        B, _, O, CO = mask.shape
        assert B == 2
        assert O == 7  # max oracle length
        assert CO == 10 + 7  # max context + max oracle

    def test_values_are_zero_or_neginf(self):
        batch = _make_batch([10, 8], [5, 7], [[0, 1], [3, 4, 5]])
        mask = create_kv_attention_mask(batch)
        is_zero = mask == 0.0
        is_neginf = mask == float("-inf")
        assert (is_zero | is_neginf).all(), "Mask contains values other than 0.0 and -inf"

    def test_dtype_propagation(self):
        batch = _make_batch([5], [3], [[1, 2]])
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            mask = create_kv_attention_mask(batch, dtype=dtype)
            assert mask.dtype == dtype


# ---------------------------------------------------------------------------
# Boundary detection and context_type tests
# ---------------------------------------------------------------------------

class _FakeOracle:
    """Minimal stand-in so we can test _find_think_end and _compute_attend_positions
    without loading a real model."""

    def __init__(self, think_end_ids: list[int]):
        self._think_end_ids = think_end_ids

    # Bind the real methods from KVCacheOracle
    _find_think_end = KVCacheOracle._find_think_end
    _compute_attend_positions = KVCacheOracle._compute_attend_positions


class TestFindThinkEnd:
    """Test _find_think_end boundary detection."""

    def test_single_token_think_end(self):
        # Simulate </think> = token 999
        oracle = _FakeOracle(think_end_ids=[999])
        full_ids = [10, 20, 30, 100, 200, 999, 300, 400]
        #           ^-- context --|  ^-- response starts at 3 --^
        result = oracle._find_think_end(full_ids, response_start=3)
        # 999 is at response_tokens index 2 (absolute index 5), so think_end = 5 + 1 = 6
        assert result == 6

    def test_multi_token_think_end(self):
        # Simulate </think> = tokens [888, 999]
        oracle = _FakeOracle(think_end_ids=[888, 999])
        full_ids = [10, 20, 30, 100, 888, 999, 300]
        result = oracle._find_think_end(full_ids, response_start=3)
        # Subsequence [888, 999] starts at response index 1 (absolute 4), len=2, so think_end = 4 + 2 = 6
        assert result == 6

    def test_no_think_tags(self):
        oracle = _FakeOracle(think_end_ids=[999])
        full_ids = [10, 20, 30, 100, 200, 300]
        result = oracle._find_think_end(full_ids, response_start=3)
        assert result is None

    def test_think_end_at_very_end(self):
        oracle = _FakeOracle(think_end_ids=[999])
        full_ids = [10, 20, 100, 200, 999]
        result = oracle._find_think_end(full_ids, response_start=2)
        assert result == 5  # right after the last token

    def test_empty_response(self):
        oracle = _FakeOracle(think_end_ids=[999])
        full_ids = [10, 20, 30]
        result = oracle._find_think_end(full_ids, response_start=3)
        assert result is None


class TestComputeAttendPositions:
    """Test _compute_attend_positions for all context_types."""

    def setup_method(self):
        self.oracle = _FakeOracle(think_end_ids=[999])

    def test_question(self):
        positions = self.oracle._compute_attend_positions(
            "question", context_token_count=10, response_start=10, response_end=20, think_end=15,
        )
        assert positions == list(range(10))

    def test_cot_with_think_tags(self):
        # CoT = response_start (10) to think_end (15)
        positions = self.oracle._compute_attend_positions(
            "cot", context_token_count=10, response_start=10, response_end=20, think_end=15,
        )
        assert positions == list(range(10, 15))

    def test_cot_without_think_tags(self):
        positions = self.oracle._compute_attend_positions(
            "cot", context_token_count=10, response_start=10, response_end=20, think_end=None,
        )
        assert positions == []

    def test_answer_with_think_tags(self):
        # Answer = think_end (15) to response_end (20)
        positions = self.oracle._compute_attend_positions(
            "answer", context_token_count=10, response_start=10, response_end=20, think_end=15,
        )
        assert positions == list(range(15, 20))

    def test_answer_without_think_tags(self):
        # No think tags: entire response is the "answer"
        positions = self.oracle._compute_attend_positions(
            "answer", context_token_count=10, response_start=10, response_end=20, think_end=None,
        )
        assert positions == list(range(10, 20))

    def test_response(self):
        positions = self.oracle._compute_attend_positions(
            "response", context_token_count=10, response_start=10, response_end=20, think_end=15,
        )
        assert positions == list(range(10, 20))

    def test_all(self):
        positions = self.oracle._compute_attend_positions(
            "all", context_token_count=10, response_start=10, response_end=20, think_end=15,
        )
        assert positions == list(range(20))

    def test_unknown_context_type_raises(self):
        with pytest.raises(ValueError, match="Unknown context_type"):
            self.oracle._compute_attend_positions(
                "invalid", context_token_count=10, response_start=10, response_end=20, think_end=15,
            )
