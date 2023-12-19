import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# > For LRA-E, however, we initialized the parameters
# > using a zero-mean Gaussian distribution (variance of 0.05).
#
# > For [...] LRA-E, we were able to use SGD (λ [step size] = 0.01).
#
# > In this paper, β = 0.1, found with only minor prelim. tuning.



class LRAENetwork(nn.Module):
    def __init__(self, input_size, num_classes, num_hiddens, hidden_size, beta=0.1, gamma=1.0, activation_fn='tanh'):
        super(LRAENetwork, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_classes = num_classes
        self.beta = beta
        self.gamma = gamma
        self.activation_fn = activation_fn

        if self.activation_fn != 'tanh':
            raise Exception('Only tanh is implemented')

        # hidden layer activation fn
        self.phi = F.tanh

        # we must store all matrices transposed w.r.t. the paper, because by convention the minibatch
        # data matrix is of shape (minibatch_size, input_size)

        # forward weights
        self.W = nn.ParameterList([nn.Parameter(torch.empty(input_size, hidden_size), requires_grad=True)]
            + [nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=True) for _ in range(num_hiddens - 1)]
            + [nn.Parameter(torch.empty(hidden_size, num_classes), requires_grad=True)])

        # self.bW = ([nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False) for _ in range(num_hiddens)]
        #            + [nn.Parameter(torch.zeros(1, num_classes), requires_grad=False)])

        # error weights, (U instead of E because I use E for the minibatch error matrix)
        self.U = nn.ParameterList([nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=True) for _ in range(num_hiddens - 1)]
            + [nn.Parameter(torch.empty(num_classes, hidden_size), requires_grad=True)])

        # self.bE = [None] + [nn.Parameter(torch.zeros(1, hidden_size), requires_grad=False) for _ in range(num_hiddens)]

        # initialize weights as in the LRA-E paper
        param_std = math.sqrt(0.05)

        # TODO: Ankur Mali's LRA-E implementation (https://github.com/AnkurMali/LRA-E/blob/master/mnist/mnist.py)
        # uses:
        #  - normal w/ stddev = 0.1 for the forward weights and zeros for forward biases
        #  - normal w/ stddev = 1 for the error weights, no error biases
        # test and compare with paper description

        for W in self.W:
            nn.init.normal_(W, mean=0.0, std=param_std)

        for U in self.U:
            if isinstance(U, nn.Parameter):
                nn.init.normal_(U, mean=0.0, std=param_std)

    # assume X is a minibatch with rows of input vectors
    def forward(self, X):
        with torch.no_grad():
            # reshape x to be (batch_size)x(input_size)
            X = X.view(X.shape[0], -1)
            self.A = []
            self.Z = []

            for (i, W) in enumerate(self.W):
                A = X @ self.W[i]# + self.bW[i]

                if i == len(self.W) - 1:
                    logits = A
                    X = F.softmax(A, dim=1)
                else:
                    X = self.phi(A)

                # Note: if using CalcUpdates-V2, you don't actually need the last h
                self.A.append(A)
                self.Z.append(X)
        return logits

    # Assumes forward() has been called. Description from paper:
    #
    # > At any given layer z^l ,
    # > starting at the output units (in our example, z3 ), we calculate
    # > the target for the layer below z^{l−1} by multiplying the error
    # > unit values at l by a set of synaptic error weights E^l . This
    # > projected displacement, weighted by the modulation factor
    # > β, is then subtracted from the initially found pre-activation
    # > of the layer below h^{l−1}
    def compute_targets(self, Y):
        # print('####################### COMPUTE_TARGETS #######################')
        # print(Y.shape)
        # print(Y[0])
        # division-by-zero-be-gone
        out_err_eps = 1e-12
        L = self.num_hiddens + 1
        self.E = [None for _ in range(L)]
        self.E[-1] = -Y / (self.Z[-1] + out_err_eps)
        # calculate targets y^k for k = L-1, ..., 1
        for k in reversed(range(L-1)):
            # U[k] instead of U[k+1] because we don't pad self.U (U^1 doesnt exist)
            Y_lower = self.phi(self.A[k] - self.beta * self.E[k+1] @ self.U[k])
            self.E[k] = 2 * (self.Z[k] - Y_lower)


    # update W and E (and bW and bE)
    def backward(self, x, y):
        m = x.shape[0]

        # one-hot encode the targets
        y_oh = F.one_hot(y, num_classes=self.num_classes)
        self.compute_targets(y_oh)

        X = x.view(x.shape[0], -1)

        # since we store weight matrices transposed w.r.t. the paper, the update rules
        # are also transposed

        self.W[0].grad = (1. / m) * X.T @ self.E[0]
        self.W[1].grad = (1. / m) * self.Z[0].T @ self.E[1]
        self.W[2].grad = (1. / m) * self.Z[1].T @ self.E[2]
        self.W[3].grad = (1. / m) * self.Z[2].T @ self.E[3]

        self.U[0].grad = -self.gamma * self.W[1].T
        self.U[1].grad = -self.gamma * self.W[2].T
        self.U[2].grad = -self.gamma * self.W[3].T

        # grad_W = [x.T @ self.E[0], self.Z[0].T @ self.E[1], self.Z[1].T @ self.E[2]]
        # grad_E = [-self.gamma * grad_W[1].T, -self.gamma * grad_W[2].T]

        # for i in range(len(grad_W)):
        #     self.W[i].grad = grad_W[i]

        # for i in range(len(grad_E)):
        #     self.U[i].grad = grad_E[i]
