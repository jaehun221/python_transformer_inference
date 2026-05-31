import numpy as np

from gpt2_decoder.config import GPT2Config
from gpt2_decoder.weights import load_weights
from gpt2_decoder.model import GPT2Model
from gpt2_decoder.attention import CausalSelfAttention


config = GPT2Config.from_json("models/gpt2/config.json")
weights = load_weights("models/gpt2/model.safetensors")

model = GPT2Model(config, weights)

# h.0 attention만 먼저 확인
attn = CausalSelfAttention.from_weights(
    config=config,
    weights=weights,
    layer_idx=0,
)

input_ids = np.array([15496, 995], dtype=np.int64)

x = model.embed(input_ids)
out = attn.forward(x)

print("input_ids.shape:", input_ids.shape)
print("x.shape:", x.shape)
print("out.shape:", out.shape)
print("out.dtype:", out.dtype)