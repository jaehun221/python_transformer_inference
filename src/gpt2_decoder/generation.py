import numpy as np

from gpt2_decoder.model import GPT2Model
from gpt2_decoder.cache import KVCache

def greedy_generate(
    model: GPT2Model,
    input_ids: np.ndarray,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> np.ndarray:
    # input_ids: [T]
    # return:    [T + num_generated_tokens]
    # num_generated_tokens <= max_new_tokens

    generated = input_ids.copy()  # [T]

    for _ in range(max_new_tokens):
        logits = model.forward(generated)  # [len(generated), vocab_size]

        next_token_logits = logits[-1]  # [vocab_size]

        next_token_id = int(np.argmax(next_token_logits))

        next_token = np.array([next_token_id], dtype=generated.dtype)  # [1]

        generated = np.concatenate([generated, next_token], axis=0)  # length + 1

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

    return generated


def greedy_generate_cached(
    model: GPT2Model,
    input_ids: np.ndarray,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> np.ndarray:
    # input_ids: [T]
    # return:    [T + num_generated_tokens]
    # num_generated_tokens <= max_new_tokens

    generated = input_ids.copy()  # [T]

    cache = KVCache(n_layer=model.config.n_layer)

    logits = model.forward_with_cache(
        input_ids=generated,
        cache=cache,
        start_pos=0,
    )  # [T, vocab_size]

    next_token_logits = logits[-1]  # [vocab_size]
    next_token_id = int(np.argmax(next_token_logits))

    for _ in range(max_new_tokens):
        next_token = np.array([next_token_id], dtype=generated.dtype)  # [1]
        generated = np.concatenate([generated, next_token], axis=0)  # length + 1

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        start_pos = cache.seq_len(0)

        logits = model.forward_with_cache(
            input_ids=next_token,
            cache=cache,
            start_pos=start_pos,
        )  # [1, vocab_size]

        next_token_logits = logits[-1]
        next_token_id = int(np.argmax(next_token_logits))

    return generated
