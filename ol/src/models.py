import torch.nn as nn

def linear_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]

def freeze_all_but_last_k(model, k=2):
    layers = linear_layers(model)
    for layer in layers[:-k]:
        for param in layer.parameters():
            param.requires_grad = False
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params <= 50000, f"Trainable parameters ({trainable_params}) exceed 50k."
    return model

# Example usage within a model context (optional, for integration)
if __name__ == "__main__":
    from src.models import MLP
    # Example instantiation
    model = MLP(in_dim=10, hidden=[128, 64], out_dim=4)
    model = freeze_all_but_last_k(model, k=2)
    print(f"Model frozen with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params.")