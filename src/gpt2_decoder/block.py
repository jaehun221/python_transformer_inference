from dataclasses import dataclass

import numpy as np

from gpt2_decoder.attention import CausalSelfAttention
from gpt2_decoder.config import GPT2Config
from gpt2_decoder.layers import LayerNorm, Linear
from gpt2_decoder.tensor import gelu_new

from gpt2_decoder.cache import KVCache

@dataclass
class MLP:
    c_fc: Linear
    c_proj: Linear

    @classmethod
    def from_weights(
        cls,
        weights: dict[str, np.ndarray],
        layer_idx: int,
    ) -> "MLP":
        prefix = f"h.{layer_idx}.mlp"

        return cls(
            c_fc=Linear(
                weight=weights[f"{prefix}.c_fc.weight"],
                bias=weights[f"{prefix}.c_fc.bias"],
            ),
            c_proj=Linear(
                weight=weights[f"{prefix}.c_proj.weight"],
                bias=weights[f"{prefix}.c_proj.bias"],
            ),
        )
    

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x:      [T, C]
        # return: [T, C]
        x = self.c_fc.forward(x)     # [T, 4C]
        x = gelu_new(x)              # [T, 4C]
        x = self.c_proj.forward(x)   # [T, C]

        return x


@dataclass
class GPT2Block:
    ln_1: LayerNorm
    attn: CausalSelfAttention
    ln_2: LayerNorm
    mlp: MLP

    @classmethod
    def from_weights(
        cls,
        config: GPT2Config,
        weights: dict[str, np.ndarray],
        layer_idx: int,
    ) -> "GPT2Block":
        prefix = f"h.{layer_idx}"

        return cls(
            ln_1=LayerNorm(
                weight=weights[f"{prefix}.ln_1.weight"],
                bias=weights[f"{prefix}.ln_1.bias"],
                eps=config.layer_norm_epsilon,
            ),
            attn=CausalSelfAttention.from_weights(
                config=config,
                weights=weights,
                layer_idx=layer_idx,
            ),
            ln_2=LayerNorm(
                weight=weights[f"{prefix}.ln_2.weight"],
                bias=weights[f"{prefix}.ln_2.bias"],
                eps=config.layer_norm_epsilon,
            ),
            mlp=MLP.from_weights(
                weights=weights,
                layer_idx=layer_idx,
            ),
        )


    def forward(
        self,
        x: np.ndarray,
        cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> np.ndarray:
        # x:      [T, C]
        # return: [T, C]

        residual = x
        x = self.ln_1.forward(x)
        x = self.attn.forward(
            x,
            cache=cache,
            layer_idx=layer_idx,
        )
        x = residual + x

        residual = x
        x = self.ln_2.forward(x)
        x = self.mlp.forward(x)
        x = residual + x

        return x