import math
from dataclasses import dataclass

import numpy as np

from gpt2_decoder.config import GPT2Config
from gpt2_decoder.layers import Linear
from gpt2_decoder.tensor import softmax


def split_heads(x: np.ndarray, n_head: int) -> np.ndarray:
    # x:      [T, C]
    # return: [H, T, D]
    T, C = x.shape
    D = C // n_head

    x = x.reshape(T, n_head, D)  # [T, H, D]
    x = x.transpose(1, 0, 2)     # [H, T, D]

    return x


def merge_heads(x: np.ndarray) -> np.ndarray:
    # x:      [H, T, D]
    # return: [T, C]
    H, T, D = x.shape

    x = x.transpose(1, 0, 2)  # [T, H, D]
    x = x.reshape(T, H * D)   # [T, C]

    return x


@dataclass
class CausalSelfAttention:
    config: GPT2Config
    c_attn: Linear
    c_proj: Linear

    @classmethod
    def from_weights(
        cls,
        config: GPT2Config,
        weights: dict[str, np.ndarray],
        layer_idx: int,
    ) -> "CausalSelfAttention":
        prefix = f"h.{layer_idx}.attn"

        return cls(
            config=config,
            c_attn=Linear(
                weights[f"{prefix}.c_attn.weight"],
                weights[f"{prefix}.c_attn.bias"],
            ),
            c_proj=Linear(
                weights[f"{prefix}.c_proj.weight"],
                weights[f"{prefix}.c_proj.bias"],
            ),
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: [T, C]
        T, C = x.shape
        H = self.config.n_head
        D = C // H

        qkv = self.c_attn.forward(x)       # [T, 3C]
        q, k, v = np.split(qkv, 3, axis=-1)  # each [T, C]

        q = split_heads(q, H)  # [H, T, D]
        k = split_heads(k, H)  # [H, T, D]
        v = split_heads(v, H)  # [H, T, D]

        scores = q @ k.transpose(0, 2, 1)  # [H, T, T]
        scores = scores / math.sqrt(D)     # [H, T, T]

        mask = np.triu(np.ones((T, T), dtype=bool), k=1)  # [T, T]
        scores = np.where(mask[None, :, :], -1e10, scores)  # [H, T, T]

        probs = softmax(scores, axis=-1)  # [H, T, T]

        out = probs @ v       # [H, T, D]
        out = merge_heads(out)  # [T, C]

        out = self.c_proj.forward(out)  # [T, C]

        return out