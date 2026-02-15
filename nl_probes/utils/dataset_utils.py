from typing import Any, Mapping

import torch
from peft import PeftModel
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule

SPECIAL_TOKEN = " ?"


def get_introspection_prefix(sae_layer: int, num_positions: int) -> str:
    prefix = f"Layer: {sae_layer}\n"
    prefix += SPECIAL_TOKEN * num_positions
    prefix += " \n"
    return prefix


class FeatureResult(BaseModel):
    """Result for a single feature evaluation."""

    feature_idx: int
    api_response: str
    prompt: str
    meta_info: Mapping[str, Any] = {}


class EvalStepResult(BaseModel):
    """Results from a single evaluation step."""

    step: int
    results: list[FeatureResult]


class TrainingDataPoint(BaseModel):
    """Training data point with tensors.
    
    Two modes of operation:
    1. Steering mode (original): steering_vectors is not None, activations are injected at positions
    2. KV cache mode: steering_vectors is None, context_input_ids/context_positions define KV attention
    
    In KV cache mode, context_positions specifies which tokens in context_input_ids the oracle
    prompt should attend to (contiguous window for single/multi-token cases)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    datapoint_type: str
    input_ids: list[int]
    labels: list[int]  # Can contain -100 for ignored tokens
    layer: int  # Layer for steering mode, -1 for KV cache mode
    steering_vectors: torch.Tensor | None
    positions: list[int]  # Positions of " ?" tokens in input_ids (for steering mode)
    feature_idx: int
    target_output: str
    context_input_ids: list[int] | None
    context_positions: list[int] | None  # Which context tokens to attend to (KV cache mode)
    ds_label: str | None  # label from the dataset
    meta_info: Mapping[str, Any] = {}

    @model_validator(mode="after")
    def _check_context_alignment(cls, values):
        sv = values.steering_vectors
        if sv is not None:
            # Steering mode: positions must match steering_vectors
            if len(values.positions) != sv.shape[0]:
                raise ValueError("positions and steering_vectors must have the same length")
        else:
            # KV cache mode: need context info
            if values.context_positions is None or values.context_input_ids is None:
                raise ValueError("context_* must be provided when steering_vectors is None")
            # In KV cache mode, positions (for " ?" tokens) and context_positions can differ
            # positions = where " ?" tokens are in the oracle prompt
            # context_positions = which context tokens to attend to
        return values


class BatchData(BaseModel):
    """Batch of training data with tensors."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor
    steering_vectors: list[torch.Tensor]
    positions: list[list[int]]
    feature_indices: list[int]


class KVCacheBatchData(BaseModel):
    """Batch of training data for KV cache mode.
    
    context_input_ids: padded context tokens [B, C_max]
    context_attention_mask: which context tokens are real (not padding) [B, C_max]
    oracle_input_ids: padded oracle prompt tokens [B, O_max]
    oracle_labels: labels for oracle tokens [B, O_max]
    oracle_attention_mask: which oracle tokens are real [B, O_max]
    context_positions: which context positions each oracle can attend to [B, list of positions]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    context_input_ids: torch.Tensor  # [B, C_max]
    context_attention_mask: torch.Tensor  # [B, C_max]
    oracle_input_ids: torch.Tensor  # [B, O_max]
    oracle_labels: torch.Tensor  # [B, O_max]
    oracle_attention_mask: torch.Tensor  # [B, O_max]
    context_positions: list[list[int]]  # Positions in context to attend to (before padding adjustment)
    context_padding_lengths: list[int]  # How much left padding was added to each context


def construct_batch(
    training_data: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> BatchData:
    max_length = 0
    for data_point in training_data:
        max_length = max(max_length, len(data_point.input_ids))

    batch_tokens = []
    batch_labels = []
    batch_attn_masks = []
    batch_positions = []
    batch_steering_vectors = []
    batch_feature_indices = []

    for data_point in training_data:
        padding_length = max_length - len(data_point.input_ids)
        padding_tokens = [tokenizer.pad_token_id] * padding_length
        padded_input_ids = padding_tokens + data_point.input_ids
        padded_labels = [-100] * padding_length + data_point.labels

        input_ids = torch.tensor(padded_input_ids, dtype=torch.long).to(device)
        labels = torch.tensor(padded_labels, dtype=torch.long).to(device)
        attn_mask = torch.ones_like(input_ids, dtype=torch.bool).to(device)

        attn_mask[:padding_length] = False

        batch_tokens.append(input_ids)
        batch_labels.append(labels)
        batch_attn_masks.append(attn_mask)

        padded_positions = [p + padding_length for p in data_point.positions]

        if data_point.steering_vectors is not None:
            steering_vectors = data_point.steering_vectors.to(device)
        else:
            steering_vectors = None

        batch_positions.append(padded_positions)
        batch_steering_vectors.append(steering_vectors)
        batch_feature_indices.append(data_point.feature_idx)

    return BatchData(
        input_ids=torch.stack(batch_tokens),
        labels=torch.stack(batch_labels),
        attention_mask=torch.stack(batch_attn_masks),
        steering_vectors=batch_steering_vectors,
        positions=batch_positions,
        feature_indices=batch_feature_indices,
    )


def construct_kv_cache_batch(
    training_data: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> KVCacheBatchData:
    """Construct a batch for KV cache training mode.
    
    Separates context (for KV caching) from oracle prompt (for forward pass).
    The oracle prompt is the full input_ids which includes the " ?" tokens and question.
    """
    # Find max lengths for padding
    max_context_len = max(len(dp.context_input_ids) for dp in training_data)
    max_oracle_len = max(len(dp.input_ids) for dp in training_data)

    context_tokens = []
    context_attn_masks = []
    oracle_tokens = []
    oracle_labels = []
    oracle_attn_masks = []
    context_positions_list = []
    context_padding_lengths = []

    pad_id = tokenizer.pad_token_id

    for dp in training_data:
        # Context: left-pad
        ctx_len = len(dp.context_input_ids)
        ctx_pad_len = max_context_len - ctx_len
        padded_ctx = [pad_id] * ctx_pad_len + list(dp.context_input_ids)
        ctx_mask = [False] * ctx_pad_len + [True] * ctx_len

        context_tokens.append(torch.tensor(padded_ctx, dtype=torch.long, device=device))
        context_attn_masks.append(torch.tensor(ctx_mask, dtype=torch.bool, device=device))
        context_padding_lengths.append(ctx_pad_len)

        # Oracle prompt: left-pad
        oracle_len = len(dp.input_ids)
        oracle_pad_len = max_oracle_len - oracle_len
        padded_oracle = [pad_id] * oracle_pad_len + dp.input_ids
        padded_labels = [-100] * oracle_pad_len + dp.labels
        oracle_mask = [False] * oracle_pad_len + [True] * oracle_len

        oracle_tokens.append(torch.tensor(padded_oracle, dtype=torch.long, device=device))
        oracle_labels.append(torch.tensor(padded_labels, dtype=torch.long, device=device))
        oracle_attn_masks.append(torch.tensor(oracle_mask, dtype=torch.bool, device=device))

        # Store original context_positions (will adjust for padding in attention mask creation)
        context_positions_list.append(list(dp.context_positions))

    return KVCacheBatchData(
        context_input_ids=torch.stack(context_tokens),
        context_attention_mask=torch.stack(context_attn_masks),
        oracle_input_ids=torch.stack(oracle_tokens),
        oracle_labels=torch.stack(oracle_labels),
        oracle_attention_mask=torch.stack(oracle_attn_masks),
        context_positions=context_positions_list,
        context_padding_lengths=context_padding_lengths,
    )


def create_kv_attention_mask(
    batch: KVCacheBatchData,
    attend_full_context: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create 4D attention mask for KV cache mode.

    Returns mask of shape [B, 1, O, C + O] where:
    - B = batch size
    - O = oracle sequence length (after padding)
    - C = context sequence length (after padding)

    The mask is in the format expected by HuggingFace models:
    - 0.0 = attend
    - -inf (large negative) = don't attend

    For each oracle token at position i:
    - Can attend to selected context positions (or all if attend_full_context=True)
    - Can attend to oracle positions 0..i (causal within oracle)
    """
    B = batch.oracle_input_ids.shape[0]
    O = batch.oracle_input_ids.shape[1]
    C = batch.context_input_ids.shape[1]

    if device is None:
        device = batch.oracle_input_ids.device

    _zero = torch.tensor(0.0, dtype=dtype, device=device)
    _neginf = torch.tensor(float("-inf"), dtype=dtype, device=device)

    # --- Region 1: Context attention [B, 1, 1, C] (broadcasts over O) ---
    if attend_full_context:
        # Attend to all non-padding context positions
        ctx_attend = batch.context_attention_mask  # [B, C] bool
    else:
        # Attend only to selected context positions (adjusted for padding)
        ctx_attend = torch.zeros(B, C, dtype=torch.bool, device=device)
        for b in range(B):
            adjusted = [p + batch.context_padding_lengths[b] for p in batch.context_positions[b]]
            if adjusted:
                ctx_attend[b, adjusted] = True

    context_region = torch.where(ctx_attend[:, None, None, :], _zero, _neginf)  # [B, 1, 1, C]

    # --- Region 2: Oracle causal self-attention [B, 1, O, O] ---
    oracle_range = torch.arange(O, device=device)
    causal = oracle_range.unsqueeze(0) <= oracle_range.unsqueeze(1)  # [O, O] lower-triangular
    oracle_real = batch.oracle_attention_mask  # [B, O] bool
    # Position i can attend to position j if: j<=i (causal), i is real, j is real
    oracle_valid = causal[None, :, :] & oracle_real[:, None, :] & oracle_real[:, :, None]  # [B, O, O]
    oracle_region = torch.where(oracle_valid[:, None, :, :], _zero, _neginf)  # [B, 1, O, O]

    # --- Combine: [B, 1, O, C + O] ---
    mask = torch.cat([context_region.expand(B, 1, O, C), oracle_region], dim=-1)

    return mask


def get_prompt_tokens_only(
    training_data_point: TrainingDataPoint,
) -> TrainingDataPoint:
    """User prompt should be labeled as -100"""
    prompt_tokens = []
    prompt_labels = []

    response_token_seen = False
    for i in range(len(training_data_point.input_ids)):
        if training_data_point.labels[i] != -100:
            response_token_seen = True
            continue
        else:
            if response_token_seen:
                raise ValueError("Response token seen before prompt tokens")
            prompt_tokens.append(training_data_point.input_ids[i])
            prompt_labels.append(training_data_point.labels[i])
    new = training_data_point.model_copy()
    new.input_ids = prompt_tokens
    new.labels = prompt_labels
    return new


def materialize_missing_steering_vectors(
    batch_points: list[TrainingDataPoint],
    tokenizer: AutoTokenizer,
    model: PeftModel,
) -> list[TrainingDataPoint]:
    """
    Materialization of missing steering vectors for a heterogenous batch
    where different items can request activations from different layers.

    Steps:
      1) Find items with steering_vectors=None.
      2) Build a left-padded batch from their context_input_ids.
      3) Register hooks for all unique requested layers and run exactly one forward pass.
      4) For each item, take activations at its requested layer and its context_positions,
         then write back a [num_positions, D] tensor to dp.steering_vectors. Returns a new batch.

    No-op if every item already has steering_vectors.
    """
    # Select datapoints that need generation
    to_fill: list[tuple[int, TrainingDataPoint]] = [
        (i, dp) for i, dp in enumerate(batch_points) if dp.steering_vectors is None
    ]
    if not to_fill:
        return batch_points

    assert isinstance(model, PeftModel), "Model must be a PeftModel"

    # Validate context fields
    for _, dp in to_fill:
        if dp.context_input_ids is None or dp.context_positions is None:
            raise ValueError(
                "Datapoint has steering_vectors=None but is missing context_input_ids or context_positions"
            )

    # Build the input batch (left padding to match your construct_batch convention)
    pad_id = tokenizer.pad_token_id
    contexts: list[list[int]] = [list(dp.context_input_ids) for _, dp in to_fill]
    positions_per_item: list[list[int]] = [list(dp.context_positions) for _, dp in to_fill]
    max_len = max(len(c) for c in contexts)

    input_ids_tensors: list[torch.Tensor] = []
    attn_masks_tensors: list[torch.Tensor] = []
    left_offsets: list[int] = []

    device = next(model.parameters()).device

    for c in contexts:
        pad_len = max_len - len(c)
        input_ids_tensors.append(torch.tensor([pad_id] * pad_len + c, dtype=torch.long, device=device))
        # For HF, bool masks are fine; your construct_batch uses bool too
        attn_masks_tensors.append(torch.tensor([False] * pad_len + [True] * len(c), dtype=torch.bool, device=device))
        left_offsets.append(pad_len)

    inputs_BL = {
        "input_ids": torch.stack(input_ids_tensors, dim=0),
        "attention_mask": torch.stack(attn_masks_tensors, dim=0),
    }

    # Prepare hooks for all unique requested layers
    layers_needed = sorted({dp.layer for _, dp in to_fill})
    submodules = {layer: get_hf_submodule(model, layer, use_lora=True) for layer in layers_needed}

    # Run a single pass with dropout off, then restore the previous train/eval mode
    was_training = model.training
    model.eval()
    with model.disable_adapter():
        # [layer] -> [B, L, D], where B == len(to_fill)
        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules=submodules,
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )
    if was_training:
        model.train()

    # Build the new list, copying only items we change
    new_batch: list[TrainingDataPoint] = list(batch_points)  # references by default
    for b in range(len(to_fill)):
        idx, dp = to_fill[b]
        layer = dp.layer
        acts_BLD = acts_by_layer[layer]  # [B, L, D] on GPU

        idxs = [p + left_offsets[b] for p in positions_per_item[b]]
        # Bounds check for safety
        L = acts_BLD.shape[1]
        if any(i < 0 or i >= L for i in idxs):
            raise IndexError(f"Activation index out of range for item {b}: {idxs} with L={L}")

        vectors = acts_BLD[b, idxs, :].detach().contiguous()

        assert len(vectors.shape) == 2, f"Expected 2D tensor, got vectors.shape={vectors.shape}"

        dp_new = dp.model_copy(deep=True)
        dp_new.steering_vectors = vectors

        new_batch[idx] = dp_new

    return new_batch


def find_pattern_in_tokens(
    token_ids: list[int], special_token_str: str, num_positions: int, tokenizer: AutoTokenizer
) -> list[int]:
    start_idx = 0
    end_idx = len(token_ids)
    special_token_id = tokenizer.encode(special_token_str, add_special_tokens=False)
    assert len(special_token_id) == 1, f"Expected single token, got {len(special_token_id)}"
    special_token_id = special_token_id[0]
    positions = []

    for i in range(start_idx, end_idx):
        if len(positions) == num_positions:
            break
        if token_ids[i] == special_token_id:
            positions.append(i)

    assert len(positions) == num_positions, f"Expected {num_positions} positions, got {len(positions)}"
    assert positions[-1] - positions[0] == num_positions - 1, f"Positions are not consecutive: {positions}"

    final_pos = positions[-1] + 1
    final_tokens = token_ids[final_pos : final_pos + 2]
    final_str = tokenizer.decode(final_tokens, skip_special_tokens=False)
    assert "\n" in final_str, f"Expected newline in {final_str}"

    return positions


def create_training_datapoint(
    datapoint_type: str,
    prompt: str,
    target_response: str,
    layer: int,
    num_positions: int,
    tokenizer: AutoTokenizer,
    acts_BD: torch.Tensor | None,
    feature_idx: int,
    context_input_ids: list[int] | None = None,
    context_positions: list[int] | None = None,
    ds_label: str | None = None,
    meta_info: Mapping[str, Any] | None = None,
) -> TrainingDataPoint:
    if meta_info is None:
        meta_info = {}
    prefix = get_introspection_prefix(layer, num_positions)
    assert prefix not in prompt, f"Prefix {prefix} found in prompt {prompt}"
    prompt = prefix + prompt
    input_messages = [{"role": "user", "content": prompt}]

    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(input_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")

    full_messages = input_messages + [{"role": "assistant", "content": target_response}]

    full_prompt_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(full_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")

    assistant_start_idx = len(input_prompt_ids)

    labels = full_prompt_ids.copy()
    for i in range(assistant_start_idx):
        labels[i] = -100

    positions = find_pattern_in_tokens(full_prompt_ids, SPECIAL_TOKEN, num_positions, tokenizer)

    if acts_BD is None:
        assert context_input_ids is not None and context_positions is not None, (
            "acts_BD is None but context_input_ids and context_positions are None"
        )
    else:
        assert len(acts_BD.shape) == 2, f"Expected 2D tensor, got {acts_BD.shape}"
        acts_BD = acts_BD.cpu().clone().detach()
        assert len(positions) == acts_BD.shape[0], f"Expected {acts_BD.shape[0]} positions, got {len(positions)}"

    training_data_point = TrainingDataPoint(
        input_ids=full_prompt_ids,
        labels=labels,
        layer=layer,
        steering_vectors=acts_BD,
        positions=positions,
        feature_idx=feature_idx,
        target_output=target_response,
        datapoint_type=datapoint_type,
        context_input_ids=context_input_ids,
        context_positions=context_positions,
        ds_label=ds_label,
        meta_info=meta_info,
    )

    return training_data_point


def create_kv_cache_training_datapoint(
    datapoint_type: str,
    prompt: str,
    target_response: str,
    tokenizer: AutoTokenizer,
    context_input_ids: list[int],
    context_positions: list[int],
    feature_idx: int = -1,
    ds_label: str | None = None,
    meta_info: Mapping[str, Any] | None = None,
) -> TrainingDataPoint:
    """Create a training datapoint for KV cache mode.
    
    Unlike steering mode, this does NOT add a "Layer: X" prefix or " ?" tokens.
    The oracle prompt is just the question, and attention to context is controlled
    via the attention mask based on context_positions.
    
    Args:
        datapoint_type: Type identifier for the datapoint
        prompt: The oracle question/prompt (no prefix added)
        target_response: Expected response
        tokenizer: Tokenizer to use
        context_input_ids: Full context tokens (will be cached as KV)
        context_positions: Which positions in context to attend to (contiguous window)
        feature_idx: Feature index (-1 for non-SAE data)
        ds_label: Dataset label
        meta_info: Additional metadata
    """
    if meta_info is None:
        meta_info = {}

    input_messages = [{"role": "user", "content": prompt}]

    input_prompt_ids = tokenizer.apply_chat_template(
        input_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(input_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")

    full_messages = input_messages + [{"role": "assistant", "content": target_response}]

    full_prompt_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
        padding=False,
        enable_thinking=False,
    )
    if not isinstance(full_prompt_ids, list):
        raise TypeError("Expected list of token ids from tokenizer")

    assistant_start_idx = len(input_prompt_ids)

    labels = full_prompt_ids.copy()
    for i in range(assistant_start_idx):
        labels[i] = -100

    # Validate context_positions are within bounds and contiguous
    if len(context_positions) > 0:
        assert all(0 <= p < len(context_input_ids) for p in context_positions), (
            f"context_positions {context_positions} out of bounds for context of length {len(context_input_ids)}"
        )
        # Check contiguity for multi-token case
        if len(context_positions) > 1:
            sorted_pos = sorted(context_positions)
            assert sorted_pos[-1] - sorted_pos[0] == len(sorted_pos) - 1, (
                f"context_positions must be contiguous, got {context_positions}"
            )

    training_data_point = TrainingDataPoint(
        input_ids=full_prompt_ids,
        labels=labels,
        layer=-1,  # KV cache mode indicator
        steering_vectors=None,
        positions=[],  # No " ?" tokens in KV cache mode
        feature_idx=feature_idx,
        target_output=target_response,
        datapoint_type=datapoint_type,
        context_input_ids=list(context_input_ids),
        context_positions=list(context_positions),
        ds_label=ds_label,
        meta_info=meta_info,
    )

    return training_data_point
