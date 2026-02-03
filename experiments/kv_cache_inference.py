"""
Minimal script to run KV cache inference with a trained LoRA model.

This demonstrates how to:
1. Load a base model + trained LoRA adapter
2. Cache context KV using the base model (no LoRA) - same as training
3. Run oracle prompt with LoRA, attending to specific context positions
4. Generate interpretations via manual decoding
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.utils.dataset_utils import (
    KVCacheBatchData,
    create_kv_attention_mask,
)


def load_model_with_lora(
    model_name: str,
    lora_path: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> tuple[AutoModelForCausalLM | PeftModel, AutoTokenizer]:
    """Load base model and optionally attach LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    )

    if lora_path is not None:
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=False)
        print(f"Loaded LoRA adapter from: {lora_path}")

    model.eval()
    return model, tokenizer


def interpret_tokens(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    context: str,
    question: str,
    token_positions: list[int] | None = None,
    attend_full_context: bool = False,
    max_new_tokens: int = 500,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """
    Interpret specific tokens in a context using the KV cache architecture.
    
    Uses the same setup as training:
    1. Context is processed with base model (LoRA disabled) to build KV cache
    2. Oracle prompt is processed with LoRA enabled, attending to cached context
    3. Generation uses manual token-by-token decoding (required for transformers 4.55+)

    Args:
        model: PeftModel with LoRA adapter
        tokenizer: Tokenizer
        context: The context text whose tokens we want to interpret
        question: The oracle question to ask about the tokens
        token_positions: Which token positions in context to attend to.
                        If None, uses last 5 tokens.
        attend_full_context: If True, attend to all context tokens
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
        dtype: Data type for attention mask

    Returns:
        Generated interpretation text
    """
    # Tokenize context
    context_tokens = tokenizer(
        context,
        return_tensors="pt",
        add_special_tokens=True,
        padding=False,
    ).to(device)

    context_input_ids = context_tokens["input_ids"]
    context_len = context_input_ids.shape[1]

    # Default: attend to last 5 tokens if not specified
    if token_positions is None:
        token_positions = list(range(max(0, context_len - 5), context_len))

    print(f"Context length: {context_len} tokens")
    print(f"Attending to positions: {token_positions}")

    # Show which tokens we're attending to
    for pos in token_positions:
        if pos < context_len:
            token_id = context_input_ids[0, pos].item()
            token_str = tokenizer.decode([token_id])
            print(f"  Position {pos}: '{token_str}'")

    # Tokenize oracle question
    oracle_messages = [{"role": "user", "content": question}]
    oracle_str = tokenizer.apply_chat_template(
        oracle_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    oracle_tokens = tokenizer(
        oracle_str,
        return_tensors="pt",
        add_special_tokens=False,
        padding=False,
    ).to(device)

    oracle_input_ids = oracle_tokens["input_ids"]
    oracle_len = oracle_input_ids.shape[1]

    # Step 1: Cache context KV using base model (without LoRA) - same as training
    with torch.no_grad():
        with model.disable_adapter():
            context_outputs = model(
                input_ids=context_input_ids,
                attention_mask=torch.ones_like(context_input_ids),
                use_cache=True,
            )
            past_key_values = context_outputs.past_key_values

    # Step 2: Create selective attention mask using the same function as training
    batch = KVCacheBatchData(
        context_input_ids=context_input_ids,
        context_attention_mask=torch.ones_like(context_input_ids, dtype=torch.bool),
        oracle_input_ids=oracle_input_ids,
        oracle_labels=torch.full_like(oracle_input_ids, -100),
        oracle_attention_mask=torch.ones_like(oracle_input_ids, dtype=torch.bool),
        context_positions=[token_positions],
        context_padding_lengths=[0],
    )

    attention_mask = create_kv_attention_mask(
        batch,
        attend_full_context=attend_full_context,
        device=device,
        dtype=dtype,
    )

    # Step 3: Process oracle prompt with LoRA and get first token prediction
    # Manual generation loop required for transformers 4.55+ with external KV cache
    with torch.no_grad():
        outputs = model(
            input_ids=oracle_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        past = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_ids = [next_token]
        
        # Generate remaining tokens
        for _ in range(max_new_tokens - 1):
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            outputs = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids.append(next_token)

    # Decode response
    generated_tokens = torch.cat(generated_ids, dim=-1)
    
    # Debug: show raw tokens
    print(f"Generated {len(generated_ids)} tokens: {generated_tokens[0].tolist()}")
    print(f"Raw decode: {tokenizer.decode(generated_tokens[0], skip_special_tokens=False)}")
    
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return response


def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen3-4B"
    LORA_PATH = "nluick/kv-cached-ao-qwen3-4b"  # Set to your HF repo path, e.g., "your-username/your-lora-repo"

    # # Example context and question
    # CONTEXT = """The capital of France is Paris. It is known for the Eiffel Tower, 
    # which was built in 1889. The city has a population of about 2 million people 
    # in the city proper and over 12 million in the metropolitan area."""

    # QUESTION = "What is this text about?"


    CONTEXT = "Englisch ist eine schöne Sprache."
    # QUESTION = "Answer with 'Yes' or 'No' only. Is this sentence written in English?"
    QUESTION = "What language is this sentence written in?"
    # Token positions to attend to (None = last 5 tokens)
    TOKEN_POSITIONS = None  # Or specify like [10, 11, 12, 13, 14]

    print("=" * 60)
    print("KV Cache Inference Demo")
    print("=" * 60)

    # Load model
    print(f"\nLoading model: {MODEL_NAME}")
    if LORA_PATH:
        print(f"With LoRA: {LORA_PATH}")
    else:
        print("Without LoRA (base model only)")

    model, tokenizer = load_model_with_lora(MODEL_NAME, LORA_PATH)

    # Check if model is a PeftModel (required for disable_adapter)
    if not isinstance(model, PeftModel):
        print("\nError: LoRA adapter required for KV cache mode.")
        print("The disable_adapter() context manager requires a PeftModel.")
        return

    print(f"\nContext:\n{CONTEXT}\n")
    print(f"Question: {QUESTION}\n")

    # Run interpretation with KV cache architecture (same as training)
    # Set attend_full_context=True to attend to all context tokens for testing
    response = interpret_tokens(
        model=model,
        tokenizer=tokenizer,
        context=CONTEXT,
        question=QUESTION,
        token_positions=TOKEN_POSITIONS,
        attend_full_context=True,  # Set to False + specific token_positions for selective attention
    )

    print(f"Response:\n{response}")
    print("=" * 60)


if __name__ == "__main__":
    main()
