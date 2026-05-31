from transformers import GPT2TokenizerFast


def load_tokenizer(model_dir: str) -> GPT2TokenizerFast:
    return GPT2TokenizerFast.from_pretrained(model_dir)