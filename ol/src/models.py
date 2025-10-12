import torch
import torch.nn as nn

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
    def __init__(self, in_dim, hidden, out_dim):
        # initialize base mlp with input, hidden, and output dims
        super().__init__()
        layers = []
        # ensure hidden is a list of integers, unpack if tuple
        hidden_list = list(hidden) if isinstance(hidden, tuple) else hidden
        dims = [in_dim] + hidden_list
        for i in range(len(dims) - 1):
            # add linear layer and relu for each hidden step
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_list[-1] if hidden_list else in_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # forward pass through sequential layers
        return self.net(x)

class MLP(BaseMLP):
    """
    Extended MLP with dropout. Use hidden sizes and activations for fixed backbone.
    """
    def __init__(self, in_dim, hidden=[512, 512], out_dim=4, dropout_p=0.0):
        # initialize mlp with optional dropout for part 3
        super().__init__(in_dim, hidden, out_dim)
        if dropout_p > 0:
            # insert dropout after each relu if enabled
            new_layers = []
            for layer in self.net:
                new_layers.append(layer)
                if isinstance(layer, nn.ReLU):
                    new_layers.append(nn.Dropout(p=dropout_p))
            self.net = nn.Sequential(*new_layers)

    def linear_layers(self):
        # extract linear layers for freezing
        return [m for m in self.modules() if isinstance(m, nn.Linear)]

    def compute_trainable_for_k(self, k: int) -> int:
        '''
        Compute trainable params for given k without freezing.
        '''
        layers = self.linear_layers()
        if k > len(layers):
            k = len(layers)  # cap at total layers
        trainable = 0
        # last k linears
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
        assert trainable_params <= limit, f"Trainable params ({trainable_params}) exceed RO limit"
        return self

# Example usage within a model context
if __name__ == "__main__":
    set_seed(4242)
    # create sample model with fixed backbone
    model = MLP(in_dim=10, hidden=[128, 64], out_dim=4)
    # freeze last 2 layers for ro
    model = model.freeze_all_but_last_k(k=2, limit=50000)
    print(f"Model frozen with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable params.")