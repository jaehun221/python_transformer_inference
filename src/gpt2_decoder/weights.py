from safetensors.numpy import load_file


# safetensors.numpy를 사용해 model.safetensors 파싱
def load_weights(path: str):
    weights = load_file(path)
    return weights

def get_tensor(weights, name):
    if name not in weights:
        raise KeyError(f"weight not found: {name}")

    return weights[name]

# safetensors 내부의 모든 weight name, shape, dtype을 출력
def print_weight_shapes(weights):
    for name in sorted(weights.keys()):
        tensor = weights[name]
        print(f"{name:50s} shape={tensor.shape} dtype={tensor.dtype}")
