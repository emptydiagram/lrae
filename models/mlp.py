import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.output_size = output_size

        self.W1 = nn.Parameter(torch.randn(input_size, hidden_size, requires_grad=True))
        self.b1 = nn.Parameter(torch.randn(hidden_size, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(hidden_size, output_size, requires_grad=True))
        self.b2 = nn.Parameter(torch.randn(output_size, requires_grad=True))

        # TODO: don't hardcode relu?

    def forward(self, x):
        # TODO: can we save some memory here?
        with torch.no_grad():
            self.A1 = x @ self.W1 + self.b1
            self.Z1 = torch.relu(self.A1)
            self.A2 = self.Z1 @ self.W2 + self.b2

        return self.A2

    def backward(self, x, y):
        # manually set gradients.
        Y = F.one_hot(y, num_classes=self.output_size)
        m = x.shape[0]
        dL_dA2 = -1.0 * (Y - self.Z2) / m
        D = -(Y - self.Z2)
        dL_dA2 = D / m
        dL_db2 = torch.mean(D, dim=0)
        dL_dW2 = self.Z1.T @ dL_dA2
        dL_dZ1 = dL_dA2 @ self.W2.T
        dL_dA1 = (dL_dZ1 * (self.A1 > 0)) / m
        dL_db1 = torch.mean(dL_dZ1 * (self.A1 > 0), dim=0)
        dL_dW1 = (x.T @ dL_dA1) / m

        self.W1.grad = dL_dW1
        self.b1.grad = dL_db1
        self.W2.grad = dL_dW2
        self.b2.grad = dL_db2