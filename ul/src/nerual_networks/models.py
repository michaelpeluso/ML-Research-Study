import torch
import torch.nn as nn
from typing import Callable, List, Tuple, Union

def set_seed(seed=4242) -> int:
    # set random seed for reproducibility per report
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

class BaseMLP(nn.Module):
    """
    Base MLP class without dropout.
    """
    def __init__(self, in_dim: int, hidden: Union[List[int], Tuple[int, ...]], out_dim: int, activation: str = 'relu') -> None:  # type: ignore
        # initialize base mlp with input, hidden, and output dims
        super().__init__()
        layers = []
        hidden_list = list(hidden) if isinstance(hidden, tuple) else hidden
        dims = [in_dim] + hidden_list
        act_fn = nn.Tanh() if activation == 'tanh' else nn.ReLU()
        for i in range(len(dims) - 1):
            # add linear layer and activation for each hidden step
            layers += [nn.Linear(dims[i], dims[i + 1]), act_fn]
        layers += [nn.Linear(hidden_list[-1] if hidden_list else in_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLP(BaseMLP):
    """
    Extended MLP with dropout. Use hidden sizes and activations for fixed backbone.
    """
    def __init__(self, in_dim: int, hidden: Union[List[int], Tuple[int, ...]] = [512, 512], out_dim: int = 4, dropout_p: float = 0.0, activation: str = 'relu') -> None:  # type: ignore
        super().__init__(in_dim, hidden, out_dim, activation=activation)  # pass activation to base
        if dropout_p > 0:
            new_layers = []
            for layer in self.net:
                new_layers.append(layer)
                if isinstance(layer, (nn.ReLU, nn.Tanh)):
                    new_layers.append(nn.Dropout(p=dropout_p))
            self.net = nn.Sequential(*new_layers)
    def linear_layers(self):
        return [m for m in self.modules() if isinstance(m, nn.Linear)]

    def compute_trainable_for_k(self, k: int) -> int:
        '''
        Compute trainable params for given k without freezing.
        '''
        layers = self.linear_layers()
        if k > len(layers):
            k = len(layers)
        trainable = 0
        for layer in layers[-k:]:
            trainable += sum(p.numel() for p in layer.parameters())
        return trainable

    def freeze_all_but_last_k(self, k=2, limit=50000):
        # freeze all but last k layers
        layers = self.linear_layers()
        for layer in layers[:-k]:
            # disable gradient updates for frozen layers
            for param in layer.parameters():
                param.requires_grad = False
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # ensure trainable params stay under 50k limit
        print(f"Froze last {k} layer(s) with {trainable_params}/{limit} trainable params")
        assert trainable_params <= limit, f"Trainable params ({trainable_params}) exceed RO limit"
        return self, trainable_params

# Example usage within a model context
if __name__ == "__main__":
    set_seed(4242)
    model = MLP(in_dim=10, hidden=[128, 64], out_dim=4)
    model, num_params = model.freeze_all_but_last_k(k=2, limit=50000)
    print(f"Model frozen with {num_params} trainable params.")