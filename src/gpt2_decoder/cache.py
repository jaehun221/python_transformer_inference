# KVCache 함수 작성
from dataclasses import dataclass, field

import numpy as np


@dataclass
class KVCache:
    # key_cache[layer_idx]:   [H, S, D]
    # value_cache[layer_idx]: [H, S, D]
    n_layer: int
    key_cache: list[np.ndarray | None] = field(init=False)
    value_cache: list[np.ndarray | None] = field(init=False)

    def __post_init__(self):
        self.key_cache = [None for _ in range(self.n_layer)]
        self.value_cache = [None for _ in range(self.n_layer)]


    def get(
        self,
        layer_idx: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        # return key/value for one layer
        # key:   None or [H, S, D]
        # value: None or [H, S, D]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


    def append(
        self,
        layer_idx: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # key:   [H, T_new, D]
        # value: [H, T_new, D]
        # return:
        # key_total:   [H, S + T_new, D]
        # value_total: [H, S + T_new, D]

        old_key, old_value = self.get(layer_idx)

        if old_key is None:
            new_key = key
            new_value = value
        else:
            new_key = np.concatenate([old_key, key], axis=1)
            new_value = np.concatenate([old_value, value], axis=1)

        self.key_cache[layer_idx] = new_key
        self.value_cache[layer_idx] = new_value

        return new_key, new_value

    def seq_len(self, layer_idx: int) -> int:
        # return cached sequence length S for one layer
        key = self.key_cache[layer_idx]

        if key is None:
            return 0

        return key.shape[1]


    def reset(self):
        # Currently unused because each generation creates a new KVCache.
        # Keep this for cases where one cache object is reused across prompts.
        for i in range(self.n_layer):
            self.key_cache[i] = None
            self.value_cache[i] = None
