import math
import numpy as np


def linear(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    # x:      [..., in_dim]
    # weight: [in_dim, out_dim]
    # bias:   [out_dim]
    # return: [..., out_dim]
    y = x @ weight

    if bias is not None:
        y = y + bias

    return y


def gelu_new(x: np.ndarray) -> np.ndarray:
    # x:      [...]
    # return: [...]
    return 0.5 * x * (
        1.0
        + np.tanh(
            math.sqrt(2.0 / math.pi)
            * (x + 0.044715 * x * x * x)
        )
    )


def layer_norm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
) -> np.ndarray:
    # x:      [..., C]
    # weight: [C]
    # bias:   [C]
    # return: [..., C]
    mean = np.mean(x, axis=-1, keepdims=True)  # [..., 1]

    centered = x - mean  # [..., C]

    var = np.mean(
        centered * centered,
        axis=-1,
        keepdims=True,
    )  # [..., 1]

    x_norm = centered / np.sqrt(var + eps)  # [..., C]

    return x_norm * weight + bias  # [..., C]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    # x:      [...]
    # return: same as x
    x_max = np.max(x, axis=axis, keepdims=True)

    exp_x = np.exp(x - x_max)

    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)