from dataclasses import dataclass

import numpy as np

from gpt2_decoder.tensor import linear, layer_norm


@dataclass
class Embedding:
    # weight: [num_embeddings, C]
    weight: np.ndarray

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        # input_ids: [T]
        # return:    [T, C]
        return self.weight[input_ids]


@dataclass
class Linear:
    # weight: [in_dim, out_dim]
    # bias:   [out_dim]
    weight: np.ndarray
    bias: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x:      [..., in_dim]
        # return: [..., out_dim]
        return linear(x, self.weight, self.bias)


@dataclass
class LayerNorm:
    # weight: [C]
    # bias:   [C]
    weight: np.ndarray
    bias: np.ndarray
    eps: float

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x:      [..., C]
        # return: [..., C]
        return layer_norm(x, self.weight, self.bias, self.eps)
