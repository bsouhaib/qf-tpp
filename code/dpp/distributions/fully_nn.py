import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from torch.distributions import constraints

from dpp.nn import BaseModule, Hypernet
from dpp.utils import clamp_preserve_gradients


class NonnegativeLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # Make weight non-negative at initialization
        self.weight.data.abs_()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        self.weight.data.clamp_(0.0)
        return F.linear(input, self.weight, self.bias)


class FullyNN(BaseModule):
    """Fully Neural Network intensity model.

    References:
        "Fully Neural Network based Model for General Temporal Point Processes",
        Omi et al., NeurIPS 2019
    """
    def __init__(self, config, n_layers=2, layer_size=64):
        super().__init__()
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.use_history(config.use_history)
        self.use_embedding(config.use_embedding)

        self.linear_time = NonnegativeLinear(1, layer_size)
        if self.using_history:
            self.linear_rnn = nn.Linear(config.history_size, layer_size, bias=False)
        if self.using_embedding:
            self.linear_emb = nn.Linear(config.embedding_size, layer_size, bias=False)

        self.linear_layers = nn.ModuleList([
            NonnegativeLinear(layer_size, layer_size) for _ in range(n_layers - 1)
        ])
        self.final_layer = NonnegativeLinear(layer_size, 1)

    def mlp(self, y, h=None, emb=None):
        y = y.unsqueeze(-1)
        hidden = self.linear_time(y)
        if h is not None:
            hidden += self.linear_rnn(h)
        if emb is not None:
            hidden += self.linear_emb(emb)
        hidden = torch.tanh(hidden)

        for linear in self.linear_layers:
            hidden = torch.tanh(linear(hidden))
        hidden = self.final_layer(hidden)
        return hidden.squeeze(-1)

    def cdf(self, y, h=None, emb=None):
        output = self.mlp(y, h, emb)
        integral = F.softplus(output)
        return -torch.expm1(-integral)
    
    def invcdf(self, tau, h=None, emb=None, delta = 1e-5):
        low, high = torch.zeros_like(tau), torch.zeros_like(tau) + 100 # TO BE CHANGED
        mid = (low + high)/2
        iterations = 0
        while torch.abs(mid - high).max() > delta:
            if iterations > 10000:
                assert(False)
            mat_bool = self.cdf(mid, h, emb) < tau
            id = torch.where(mat_bool)
            id2 = torch.where(~mat_bool)
            low[id] = mid[id]
            high[id2] = mid[id2]
            mid = (low + high)/2
            iterations = iterations + 1
        return mid
        # avoid the max in every iteration. Instead focus on those we have not converged yet
        # id = torch.where(torch.abs(mid - high) > delta)

    
    def sample(self, n_samples, h=None, emb=None):

        if (h is not None):
            first_dims = h.shape[:-1]
        elif (emb is not None):
            first_dims = emb.shape[:-1]
        else:
            first_dims = torch.Size()
        shape = first_dims + torch.Size([n_samples])

        dist = td.uniform.Uniform(0, 1)
        taus = dist.rsample(shape)

        samples = torch.zeros([h.shape[0], h.shape[1], n_samples])
        #taus = torch.rand(n_samples)
        for i in range(n_samples):
            samples[:, :, i] = self.invcdf(taus[:, :, i], h, emb)
        return samples


    def log_cdf(self, y, h=None, emb=None):
        return torch.log(self.cdf(y, h, emb) + 1e-8)

    def log_prob(self, y, h=None, emb=None):
        y.requires_grad_()
        output = self.mlp(y, h, emb)
        integral = F.softplus(output)
        intensity = torch.autograd.grad(integral, y, torch.ones_like(output), create_graph=True)[0]
        log_p = torch.log(intensity + 1e-8) - integral
        return log_p
