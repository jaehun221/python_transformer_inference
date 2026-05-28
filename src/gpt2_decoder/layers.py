from dataclasses import dataclass

import numpy as np

from gpt2_decoder.tensor import linear, layer_norm


@dataclass
class Embedding:
    weight: np.ndarray

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        return self.weight[input_ids]


@dataclass
class Linear:
    weight: np.ndarray
    bias: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        return linear(x, self.weight, self.bias)


@dataclass
class LayerNorm:
    weight: np.ndarray
    bias: np.ndarray
    eps: float

    def forward(self, x: np.ndarray) -> np.ndarray:
        return layer_norm(x, self.weight, self.bias, self.eps)