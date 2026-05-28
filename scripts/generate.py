from gpt2_decoder.config import GPT2Config

config = GPT2Config.from_json("models/gpt2/config.json")

print(config)