"""
KV Cache Oracle for inference with trained LoRA models.

Supports attending to different parts of the target model's output:
- "question": attend to input question tokens only (default)
- "cot": attend to target model's Chain-of-Thought tokens (<think>...</think>)
- "answer": attend to target model's final answer tokens (after </think>)
- "response": attend to full response (CoT + answer)
- "all": attend to everything (question + response)

Usage:
    oracle = KVCacheOracle("Qwen/Qwen3-4B", "nluick/kv-cached-ao-qwen3-4b")
    result = oracle.interpret("Some text", "What language is this?", context_type="question")
"""

import logging
from typing import Literal

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.utils.dataset_utils import KVCacheBatchData, create_kv_attention_mask

logger = logging.getLogger(__name__)

ContextType = Literal["question", "cot", "answer", "response", "all"]


class KVCacheOracle:
    """Persistent KV-cache oracle for inference. Load once, call interpret() many times."""

    def __init__(
        self,
        model_name: str,
        lora_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        """Load base model + LoRA adapter once.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen3-4B").
            lora_path: Path or HF repo for the trained LoRA adapter.
            dtype: Model dtype.
            device: Device to load model on.
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device,
        )
        self.model = PeftModel.from_pretrained(self.model, lora_path, is_trainable=False)
        self.model.eval()

        # Pre-compute think tag token IDs for boundary detection
        self._think_end_ids: list[int] = self.tokenizer.encode("</think>", add_special_tokens=False)

        # Collect all stop token IDs for generation (EOS + any model-specific end tokens)
        self._stop_ids: set[int] = {self.tokenizer.eos_token_id}
        if hasattr(self.tokenizer, "added_tokens_encoder"):
            for tok_str, tok_id in self.tokenizer.added_tokens_encoder.items():
                if "end" in tok_str.lower() or tok_str in ("<|im_end|>", "<|eot_id|>"):
                    self._stop_ids.add(tok_id)

    def interpret(
        self,
        context: str,
        question: str,
        context_type: ContextType = "question",
        token_positions: list[int] | None = None,
        attend_full_context: bool = False,
        target_response: str | None = None,
        max_new_tokens: int = 500,
        generate_max_new_tokens: int = 2048,
        do_sample: bool = False,
        temperature: float = 1.0,
        suppress_stop_for: int = 0,
        verbose: bool = False,
    ) -> str:
        """Run oracle inference on a single input.

        Args:
            context: The input text (question/prompt for the target model).
            question: The oracle's question about the context.
            context_type: Which part of context+response to attend to.
                - "question": attend only to input question tokens (default).
                - "cot": attend to Chain-of-Thought tokens (inside <think>...</think>).
                - "answer": attend to final answer tokens (after </think>).
                - "response": attend to full response (CoT + answer).
                - "all": attend to everything (question + response).
            token_positions: Explicit token positions to attend to (overrides context_type).
            attend_full_context: If True AND context_type is "question" AND token_positions
                is None, attend to ALL question tokens. Ignored for other context_types.
            target_response: Pre-generated response text. If None and context_type needs
                response tokens, the base model generates one live.
            max_new_tokens: Maximum tokens for the oracle to generate.
            generate_max_new_tokens: Maximum tokens for target model response generation.
            do_sample: Whether to use sampling for oracle generation.
            temperature: Temperature for sampling.
            suppress_stop_for: Number of initial tokens during which stop/special tokens
                are suppressed (logits set to -inf). Useful when the oracle immediately
                emits a stop token for unfamiliar context types.
            verbose: If True, print diagnostic info (token counts, attend positions,
                first generated token) to help debug empty or unexpected responses.

        Returns:
            Generated oracle interpretation text.
        """
        needs_response = context_type in ("cot", "answer", "response", "all")

        # 1. Tokenize context with chat template (aligns with training convention)
        context_messages = [{"role": "user", "content": context}]
        context_str = self.tokenizer.apply_chat_template(
            context_messages, tokenize=False, add_generation_prompt=needs_response,
        )
        context_token_ids = self.tokenizer.encode(context_str, add_special_tokens=False)
        context_token_count = len(context_token_ids)

        # 2. Build full context (possibly including target model response)
        if needs_response:
            if target_response is None:
                # Live generation: returns raw token IDs (already stripped of trailing stop tokens)
                response_token_ids = self._generate_target_response(
                    context, max_new_tokens=generate_max_new_tokens,
                )
            else:
                # Pre-generated string: encode and strip trailing stop tokens
                response_token_ids = self.tokenizer.encode(target_response, add_special_tokens=False)
                special_ids = self.tokenizer.all_special_ids
                while response_token_ids and response_token_ids[-1] in special_ids:
                    response_token_ids.pop()
            full_context_ids = context_token_ids + response_token_ids
        else:
            response_token_ids = []
            full_context_ids = context_token_ids

        if verbose:
            print(f"  [diag] context_type={context_type!r}")
            print(f"  [diag] context_tokens={context_token_count}, response_tokens={len(response_token_ids)}, total={len(full_context_ids)}")
            if response_token_ids:
                decoded_resp = self.tokenizer.decode(response_token_ids, skip_special_tokens=False)
                print(f"  [diag] response preview: {decoded_resp[:200]}{'...' if len(decoded_resp) > 200 else ''}")

        # 3. Compute which positions to attend to
        if token_positions is not None:
            # Explicit positions override context_type
            attend_positions = token_positions
            use_attend_full = False
        elif context_type == "question":
            if attend_full_context:
                attend_positions = list(range(context_token_count))
                use_attend_full = True
            else:
                # Default: last 5 tokens of the question
                attend_positions = list(range(max(0, context_token_count - 5), context_token_count))
                use_attend_full = False
        else:
            # Compute positions from response boundaries
            response_start = context_token_count
            response_end = len(full_context_ids)
            think_end = self._find_think_end(full_context_ids, response_start)
            attend_positions = self._compute_attend_positions(
                context_type, context_token_count, response_start, response_end, think_end,
            )
            use_attend_full = False

        if verbose:
            print(f"  [diag] attend_positions count={len(attend_positions)}", end="")
            if attend_positions:
                print(f", range=[{min(attend_positions)}..{max(attend_positions)}]", end="")
                # Show what tokens are being attended to
                attended_text = self.tokenizer.decode(
                    [full_context_ids[p] for p in attend_positions if p < len(full_context_ids)],
                    skip_special_tokens=False,
                )
                print(f"\n  [diag] attended text: {attended_text[:200]}{'...' if len(attended_text) > 200 else ''}")
            else:
                print()

        if not attend_positions:
            logger.warning(
                "No positions to attend to for context_type=%r. "
                "The oracle will receive no context signal.",
                context_type,
            )

        # 4. Cache context KV with base model (LoRA disabled)
        full_context_tensor = torch.tensor([full_context_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            with self.model.disable_adapter():
                context_outputs = self.model(
                    input_ids=full_context_tensor,
                    attention_mask=torch.ones_like(full_context_tensor),
                    use_cache=True,
                )
                past_key_values = context_outputs.past_key_values

        # 5. Tokenize oracle question
        oracle_messages = [{"role": "user", "content": question}]
        oracle_str = self.tokenizer.apply_chat_template(
            oracle_messages, tokenize=False, add_generation_prompt=True,
        )
        oracle_ids = self.tokenizer.encode(oracle_str, add_special_tokens=False)
        oracle_tensor = torch.tensor([oracle_ids], dtype=torch.long, device=self.device)

        # 6. Build attention mask
        batch = KVCacheBatchData(
            context_input_ids=full_context_tensor,
            context_attention_mask=torch.ones_like(full_context_tensor, dtype=torch.bool),
            oracle_input_ids=oracle_tensor,
            oracle_labels=torch.full_like(oracle_tensor, -100),
            oracle_attention_mask=torch.ones_like(oracle_tensor, dtype=torch.bool),
            context_positions=[attend_positions],
            context_padding_lengths=[0],
        )

        attention_mask = create_kv_attention_mask(
            batch, attend_full_context=use_attend_full, device=self.device, dtype=self.dtype,
        )

        # 7. Compute oracle position IDs to match training RoPE distances.
        # During training, the oracle follows immediately after a short context (~15 tokens).
        # For response-type contexts, the KV cache is much longer (~200-400 tokens), pushing
        # the oracle far away in position space. We fix this by placing the oracle right after
        # the last attended position, so relative distances match training.
        if attend_positions:
            oracle_start_pos = max(attend_positions) + 1
        else:
            oracle_start_pos = len(full_context_ids)

        oracle_seq_len = oracle_tensor.shape[1]
        oracle_position_ids = torch.arange(
            oracle_start_pos, oracle_start_pos + oracle_seq_len, device=self.device,
        ).unsqueeze(0)

        if verbose:
            default_start = len(full_context_ids)
            print(
                f"  [diag] oracle position_ids start={oracle_start_pos} "
                f"(default would be {default_start}, delta={default_start - oracle_start_pos})"
            )

        # 8. Generate oracle response
        # model.generate() rejects 4D attention masks in transformers 4.55+,
        # so we use a manual prefill + decode loop.
        suppress_ids = list(self._stop_ids) if suppress_stop_for > 0 else []

        def _sample(logits: torch.Tensor, step: int) -> torch.Tensor:
            """Pick next token from logits, optionally suppressing stop tokens."""
            if step < suppress_stop_for and suppress_ids:
                logits = logits.clone()
                logits[:, suppress_ids] = float("-inf")
            if do_sample and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                return torch.multinomial(probs, num_samples=1)
            return logits.argmax(dim=-1, keepdim=True)

        with torch.no_grad():
            # Prefill: process the full oracle prompt with the 4D KV-cache mask
            outputs = self.model(
                input_ids=oracle_tensor,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=oracle_position_ids,
                use_cache=True,
            )
            past = outputs.past_key_values
            next_token = _sample(outputs.logits[:, -1, :], step=0)
            generated = [next_token]

            if verbose:
                first_id = next_token.item()
                first_str = self.tokenizer.decode([first_id], skip_special_tokens=False)
                is_stop = first_id in self._stop_ids
                print(f"  [diag] first generated token: id={first_id}, text={first_str!r}, is_stop={is_stop}")
                if is_stop and suppress_stop_for == 0:
                    print(f"  [diag] TIP: try suppress_stop_for=1 (or higher) to force past initial stop tokens.")

            # Decode loop: generate one token at a time
            decode_pos = oracle_start_pos + oracle_seq_len
            for step in range(1, max_new_tokens):
                if next_token.item() in self._stop_ids and step >= suppress_stop_for:
                    break
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past,
                    position_ids=torch.tensor([[decode_pos]], device=self.device),
                    use_cache=True,
                )
                decode_pos += 1
                past = outputs.past_key_values
                next_token = _sample(outputs.logits[:, -1, :], step=step)
                generated.append(next_token)

        generated_tokens = torch.cat(generated, dim=-1)
        return self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    def _generate_target_response(self, context: str, max_new_tokens: int = 2048) -> list[int]:
        """Generate response token IDs from the base model (LoRA disabled).

        Returns raw token IDs with trailing stop tokens stripped, preserving
        internal special tokens like <think>/<​/think> for boundary detection.
        """
        messages = [{"role": "user", "content": context}]
        input_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer.encode(input_str, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            with self.model.disable_adapter():
                output_ids = self.model.generate(
                    input_ids=input_tensor,
                    attention_mask=torch.ones_like(input_tensor),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        generated: list[int] = output_ids[0, input_tensor.shape[1]:].tolist()

        # Strip trailing stop/special tokens (EOS, <|im_end|>, etc.)
        # but keep internal ones like <think>/<​/think>
        special_ids = self.tokenizer.all_special_ids
        while generated and generated[-1] in special_ids:
            generated.pop()

        return generated

    def _find_think_end(self, full_token_ids: list[int], response_start: int) -> int | None:
        """Find the position right after </think> in the response portion.

        Returns:
            Absolute index in full_token_ids right after the </think> tag,
            or None if no </think> tag is found.
        """
        tag = self._think_end_ids
        tag_len = len(tag)
        response_tokens = full_token_ids[response_start:]

        for i in range(len(response_tokens) - tag_len + 1):
            if response_tokens[i : i + tag_len] == tag:
                return response_start + i + tag_len

        return None

    def _compute_attend_positions(
        self,
        context_type: ContextType,
        context_token_count: int,
        response_start: int,
        response_end: int,
        think_end: int | None,
    ) -> list[int]:
        """Compute which token positions to attend to based on context_type."""
        if context_type == "question":
            return list(range(context_token_count))

        if context_type == "cot":
            if think_end is None:
                logger.warning("No <think>...</think> tags found. context_type='cot' returns empty positions.")
                return []
            return list(range(response_start, think_end))

        if context_type == "answer":
            if think_end is None:
                # No thinking tags: entire response is the "answer"
                return list(range(response_start, response_end))
            return list(range(think_end, response_end))

        if context_type == "response":
            return list(range(response_start, response_end))

        if context_type == "all":
            return list(range(response_end))

        raise ValueError(f"Unknown context_type: {context_type!r}")


if __name__ == "__main__":
    oracle = KVCacheOracle(
        model_name="Qwen/Qwen3-4B",
        lora_path="nluick/kv-cached-ao-qwen3-4b",
    )

    # Example 1: Attend to all question tokens (current behavior)
    print("=" * 60)
    print("Example 1: question-only, attend_full_context=True")
    result = oracle.interpret(
        context="Englisch ist eine schöne Sprache.",
        question="What language is this sentence written in?",
        context_type="question",
        attend_full_context=True,
        verbose=True,
    )
    print(f"Response: {result}")

    # Example 2: Attend to target model's CoT (live generation)
    print("=" * 60)
    print("Example 2: attend to model's CoT")
    result = oracle.interpret(
        context="What is 2+2?",
        question="Is the model's reasoning correct? Answer Yes or No.",
        context_type="cot",
        suppress_stop_for=1,
        verbose=True,
    )
    print(f"Response: {result}")

    # Example 3: Attend to pre-generated answer
    print("=" * 60)
    print("Example 3: attend to pre-generated answer")
    result = oracle.interpret(
        context="What is the capital of France?",
        question="Is this answer correct? Answer Yes or No.",
        context_type="answer",
        target_response="<think>Let me think... France is a country in Europe.</think>The capital is Paris.",
        suppress_stop_for=1,
        verbose=True,
    )
    print(f"Response: {result}")

    # Example 4: Attend to full response (CoT + answer)
    print("=" * 60)
    print("Example 4: attend to full response")
    result = oracle.interpret(
        context="Translate 'hello' to French.",
        question="What task was the model performing?",
        context_type="response",
        suppress_stop_for=1,
        verbose=True,
    )
    print(f"Response: {result}")
