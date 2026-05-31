import numpy as np

from gpt2_decoder.model import GPT2Model


def greedy_generate(
    model: GPT2Model,
    input_ids: np.ndarray,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> np.ndarray:
    # input_ids: [T]
    # return:    [T + max_new_tokens] or shorter if eos appears

    generated = input_ids.copy()  # [T]

    for _ in range(max_new_tokens):
        logits = model.forward(generated)  # [cur_T, vocab_size]

        next_token_logits = logits[-1]  # [vocab_size]

        next_token_id = int(np.argmax(next_token_logits))

        next_token = np.array([next_token_id], dtype=generated.dtype)  # [1]

        generated = np.concatenate([generated, next_token], axis=0)  # [cur_T + 1]

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

    return generated