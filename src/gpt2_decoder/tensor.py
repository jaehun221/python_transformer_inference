import math
import numpy as np


def linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    y = x @ weight

    if bias is not None:
        y = y + bias

    return y



def gelu_new(x: np.ndarray) -> np.ndarray:
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
    mean = np.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    
    # 분산이 매우 작아서 0으로 나누는 경우를 막기 위해 아주 작은 값 eps를 더함
    var = np.mean(centered * centered, axis=-1, keepdims=True)
    x_norm = centered / np.sqrt(var + eps) 

    return x_norm * weight + bias



def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
