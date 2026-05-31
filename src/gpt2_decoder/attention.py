import math
from dataclasses import dataclass

import numpy as np

from gpt2_decoder.cache import KVCache
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
                weight=weights[f"{prefix}.c_attn.weight"],
                bias=weights[f"{prefix}.c_attn.bias"],
            ),
            c_proj=Linear(
                weight=weights[f"{prefix}.c_proj.weight"],
                bias=weights[f"{prefix}.c_proj.bias"],
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
        T, C = x.shape
        H = self.config.n_head
        D = C // H

        qkv = self.c_attn.forward(x)  # [T, 3C]

        q, k, v = np.split(qkv, 3, axis=-1)  # each [T, C]

        q = split_heads(q, H)  # [H, T, D]
        k = split_heads(k, H)  # [H, T, D]
        v = split_heads(v, H)  # [H, T, D]

        if cache is not None:
            if layer_idx is None:
                raise ValueError("layer_idx must be provided when cache is used")

            # k, v:          [H, T, D]
            # cached k, v:   [H, S + T, D]
            k, v = cache.append(
                layer_idx=layer_idx,
                key=k,
                value=v,
            )

            # cached mode:
            # q: [H, T, D]
            # k: [H, S + T, D]
            key_len = k.shape[1]
        else:
            # no-cache mode:
            # q: [H, T, D]
            # k: [H, T, D]
            key_len = T

        scores = q @ k.transpose(0, 2, 1)  # [H, T, key_len]
        scores = scores / math.sqrt(D)     # [H, T, key_len]

        if cache is None:
            # no-cache full sequence attention.
            # token i cannot attend to future token j > i.
            mask = np.triu(
                np.ones((T, T), dtype=bool),
                k=1,
            )  # [T, T]

            scores = np.where(
                mask[None, :, :],
                -1e10,
                scores,
            )  # [H, T, T]
        else:
            # cached generation에서는 보통 decode step에서 T=1이다.
            # 이미 cache에는 과거 K/V + 현재 K/V가 들어 있으므로
            # 현재 query는 key_len 전체를 볼 수 있다.
            #
            # 단, prefill을 cache mode로 여러 token 한 번에 넣는 경우에는
            # 같은 chunk 안의 future token을 막기 위한 causal mask가 필요하다.
            past_len = key_len - T

            query_positions = past_len + np.arange(T)      # [T]
            key_positions = np.arange(key_len)             # [key_len]

            mask = key_positions[None, :] > query_positions[:, None]  # [T, key_len]

            scores = np.where(
                mask[None, :, :],
                -1e10,
                scores,
            )  # [H, T, key_len]

        probs = softmax(scores, axis=-1)  # [H, T, key_len]

        out = probs @ v         # [H, T, D]
        out = merge_heads(out)  # [T, C]

        out = self.c_proj.forward(out)  # [T, C]

        return out