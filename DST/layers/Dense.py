from torch import nn
class Dense(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None, device='cpu'):
        super(Dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias).to(device) # Linear layer
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x