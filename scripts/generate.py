import numpy as np

from gpt2_decoder.config import GPT2Config
from gpt2_decoder.generation import greedy_generate, greedy_generate_cached
from gpt2_decoder.model import GPT2Model
from gpt2_decoder.tokenizer import load_tokenizer
from gpt2_decoder.weights import load_weights


MODEL_DIR = "models/gpt2"


def main():
    config = GPT2Config.from_json(f"{MODEL_DIR}/config.json")
    weights = load_weights(f"{MODEL_DIR}/model.safetensors")

    model = GPT2Model(config, weights)
    tokenizer = load_tokenizer(MODEL_DIR)

    prompt = "Hello"

    input_ids = tokenizer.encode(prompt, return_tensors=None)
    input_ids = np.array(input_ids, dtype=np.int64)  # [T]

    output_ids_no_cache = greedy_generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=20,
        eos_token_id=config.eos_token_id,
    )  # [T + new_T]

    output_ids_cache = greedy_generate_cached(
        model=model,
        input_ids=input_ids,
        max_new_tokens=20,
        eos_token_id=config.eos_token_id,
    )

    print("no_cache:", output_ids_no_cache)
    print("cache:   ", output_ids_cache)
    print("same:", np.array_equal(output_ids_no_cache, output_ids_cache))

    text = tokenizer.decode(output_ids_cache.tolist())

    print("text:")
    print(text)

    

if __name__ == "__main__":
    main()