import numpy as np

from gpt2_decoder.block import GPT2Block
from gpt2_decoder.config import GPT2Config
from gpt2_decoder.layers import Embedding, LayerNorm


class GPT2Model:
    def __init__(self, config: GPT2Config, weights: dict[str, np.ndarray]):
        self.config = config

        # wte.weight: [vocab_size, C]
        self.wte = Embedding(weights["wte.weight"])

        # wpe.weight: [n_positions, C]
        self.wpe = Embedding(weights["wpe.weight"])

        # h.0 ~ h.11
        self.blocks = [
            GPT2Block.from_weights(
                config=config,
                weights=weights,
                layer_idx=i,
            )
            for i in range(config.n_layer)
        ]

        # ln_f.weight: [C]
        # ln_f.bias:   [C]
        self.ln_f = LayerNorm(
            weight=weights["ln_f.weight"],
            bias=weights["ln_f.bias"],
            eps=config.layer_norm_epsilon,
        )

    def embed(self, input_ids: np.ndarray) -> np.ndarray:
        # input_ids: [T]
        # return:    [T, C]
        seq_len = input_ids.shape[0]

        position_ids = np.arange(seq_len, dtype=np.int64)  # [T]

        token_embeds = self.wte.forward(input_ids)         # [T, C]
        position_embeds = self.wpe.forward(position_ids)   # [T, C]

        x = token_embeds + position_embeds                 # [T, C]

        return x

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        # input_ids: [T]
        # return:    [T, vocab_size]
        x = self.embed(input_ids)  # [T, C]

        for block in self.blocks:
            x = block.forward(x)   # [T, C]

        x = self.ln_f.forward(x)   # [T, C]

        # GPT-2는 token embedding weight를 lm head로 재사용한다.
        # wte.weight:   [vocab_size, C]
        # wte.weight.T: [C, vocab_size]
        logits = x @ self.wte.weight.T  # [T, vocab_size]

        return logits