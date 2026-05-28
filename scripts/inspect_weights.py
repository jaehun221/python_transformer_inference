from gpt2_decoder.weights import load_weights, print_weight_shapes


weights = load_weights("models/gpt2/model.safetensors")
print(f"number of tensors: {len(weights)}")
print_weight_shapes(weights)