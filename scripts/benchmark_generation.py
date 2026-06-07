import time

import numpy as np

from gpt2_decoder.config import GPT2Config
from gpt2_decoder.generation import greedy_generate, greedy_generate_cached
from gpt2_decoder.model import GPT2Model
from gpt2_decoder.tokenizer import load_tokenizer
from gpt2_decoder.weights import load_weights


MODEL_DIR = "models/gpt2"
PROMPTS = [
    ("short", "Hello"),
    ("medium", "The quick brown fox jumps over the lazy dog."),
    (
        "long",
        "In this project, I implemented a GPT-2 decoder-only Transformer "
        "using NumPy to understand how token embeddings, causal self-attention, "
        "MLP blocks, residual connections, and KV cache work together during "
        "autoregressive text generation.",
    ),
]
MAX_NEW_TOKENS_LIST = [10, 30, 50]
WARMUP_REPEATS = 1
REPEATS = 5


def measure(
    fn,
    warmup_repeats: int,
    repeats: int,
) -> tuple[float, np.ndarray]:
    times = []
    output = None

    for _ in range(warmup_repeats):
        fn()

    for _ in range(repeats):
        start = time.perf_counter()
        output = fn()
        times.append(time.perf_counter() - start)

    assert output is not None

    return sum(times) / len(times), output


def main():
    config = GPT2Config.from_json(f"{MODEL_DIR}/config.json")
    weights = load_weights(f"{MODEL_DIR}/model.safetensors")

    model = GPT2Model(config, weights)
    tokenizer = load_tokenizer(MODEL_DIR)

    print(f"warmup repeats: {WARMUP_REPEATS}")
    print(f"repeats: {REPEATS}")
    print()
    print("| prompt | prompt_tokens | max_new_tokens | no_cache | cache | speedup | same |")
    print("|---|---:|---:|---:|---:|---:|:---:|")

    for prompt_name, prompt in PROMPTS:
        input_ids = tokenizer.encode(prompt, return_tensors=None)
        input_ids = np.array(input_ids, dtype=np.int64)  # [T]
        prompt_tokens = input_ids.shape[0]

        for max_new_tokens in MAX_NEW_TOKENS_LIST:
            no_cache_time, no_cache_output = measure(
                lambda: greedy_generate(
                    model=model,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=None,
                ),
                warmup_repeats=WARMUP_REPEATS,
                repeats=REPEATS,
            )

            cache_time, cache_output = measure(
                lambda: greedy_generate_cached(
                    model=model,
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=None,
                ),
                warmup_repeats=WARMUP_REPEATS,
                repeats=REPEATS,
            )

            same = np.array_equal(no_cache_output, cache_output)
            speedup = no_cache_time / cache_time

            print(
                f"| {prompt_name} "
                f"| {prompt_tokens} "
                f"| {max_new_tokens} "
                f"| {no_cache_time:.4f}s "
                f"| {cache_time:.4f}s "
                f"| {speedup:.2f}x "
                f"| {same} |"
            )


if __name__ == "__main__":
    main()
