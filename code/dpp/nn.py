import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np


class BaseModule(nn.Module):
    """Wrapper around nn.Module that recursively sets history and embedding usage.

    All modules should inherit from this class.
    """
    def __init__(self):
        super().__init__()
        self._using_history = False
        self._using_embedding = False
        self._using_marks = False

    @property
    def using_history(self):
        return self._using_history

    @property
    def using_embedding(self):
        return self._using_embedding

    @property
    def using_marks(self):
        return self._using_marks

    def use_history(self, mode=True):
        """Recursively make all submodules use history."""
        self._using_history = mode
        for module in self.children():
            if isinstance(module, BaseModule):
                module.use_history(mode)

    def use_embedding(self, mode=True):
        """Recursively make all submodules use embeddings."""
        self._using_embedding = mode
        for module in self.children():
            if isinstance(module, BaseModule):
                module.use_embedding(mode)

    def use_marks(self, mode=True):
        """Recursively make all submodules use embeddings."""
        self._using_marks = mode
        for module in self.children():
            if isinstance(module, BaseModule):
                module.use_marks(mode)

    def sample(self, n_samples, h=None, emb=None, params = None):
        if params is None:
            params = self.get_params(h, emb)

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
        for i in range(n_samples):
            samples[:, :, i] = self.invcdf(taus[:, :, i], params = params)
        
        return samples

    def crps_sapprox(self, y, h=None, emb=None, N = 100, return_details = True):

        samples = self.sample(N, h, emb).detach()
        
        ##################@
        # Other metrics
        n = 100 # 200
        probs = torch.arange(1,n)/n

        qf = torch.quantile(samples, probs, dim = 2).permute([1, 2, 0])

        ## ACE
        abs_calibration_error = torch.abs( probs.unsqueeze(0) - torch.sum(y.unsqueeze(-1) < qf, axis = 1)/ y.shape[1])

        # Target interval coverage
        c = int(n/2)
        target_interval_coverage = (probs[np.arange(c, n-1)] - probs[np.arange(c-2, -1, -1)])

        pi_coverage_prob, interval_scores_list = [], []
        # Multiple prediction intervals
        for k in np.arange(1, c):
            i_l = (c-1) - k 
            i_u = (c-1) + k

            q_L = qf[:, :, i_l].squeeze(-1) # ERROR HERE !!!
            q_U = qf[:, :, i_u].squeeze(-1)
            

            # Interval Score
            alpha = 1 - (probs[i_u] - probs[i_l])
            interval_scores_k = (q_U - q_L)  + (2.0 / alpha) * (q_L - y) * (y <= q_L)  + (2.0 / alpha) * (y - q_U ) * (y >= q_U) 
            
            #interval_scores_avg = torch.mean(interval_scores_k).item()
            
            interval_scores_list.append(interval_scores_k)

            # PICP
            pi_coverage_prob_k = ((y >= q_L) *  (y <= q_U)).sum()/y.shape[1]
            pi_coverage_prob.append(pi_coverage_prob_k.item())
        
        interval_scores = torch.stack(interval_scores_list, 2)
        mean_abs_calibration_error =  np.mean([np.abs(tic - picp) for tic, picp in zip(target_interval_coverage, pi_coverage_prob)])        
        ##################@

        sample_sorted, _  = torch.sort(samples, dim = 2)
        
        indices = torch.arange(1, N+1)
        
        check_y = y.unsqueeze(-1) < sample_sorted
        qs = (2.0/N) * (sample_sorted - y.unsqueeze(-1)) * (N * check_y - indices.unsqueeze(0) + 0.5)
        
        if return_details:
            means = torch.mean(samples, -1)
            medians = torch.median(samples, -1)[0]

            results = torch.mean(qs, dim = 2), qs, (torch.sum(~check_y, dim=-1) + 1)/(N+1), means, torch.std(samples, -1), (means - y).pow(2), torch.abs(medians - y)
            results = results + (target_interval_coverage, abs_calibration_error, mean_abs_calibration_error, pi_coverage_prob)
            return results

        else:
            return torch.mean(qs, dim = 2)
    
class RNNLayer(BaseModule):
    """RNN for encoding the event history."""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.history_size
        self.rnn_type = config.rnn_type
        self.use_history(config.use_history)
        self.use_marks(config.use_marks)

        if config.use_marks:
            # Define mark embedding layer
            self.mark_embedding = nn.Embedding(config.num_classes, config.mark_embedding_size)
            # If we have marks, input is time + mark embedding vector
            self.in_features = config.mark_embedding_size + 1
        else:
            # Without marks, input is only time
            self.in_features = 1

        # Possible RNN types: 'RNN', 'GRU', 'LSTM'
        self.rnn = getattr(nn, self.rnn_type)(self.in_features, self.hidden_size, batch_first=True)

    def forward(self, input):
        """Encode the history of the given batch.

        Returns:
            h: History encoding, shape (batch_size, seq_len, self.hidden_size)
        """
        t = input.in_time
        length = input.length

        if not self.using_history:
            return torch.zeros(t.shape[0], t.shape[1], self.hidden_size)

        x = t.unsqueeze(-1)
        if self.using_marks:
            mark = self.mark_embedding(input.in_mark)
            x = torch.cat([x, mark], -1)

        h_shape = (1, x.shape[0], self.hidden_size)
        if self.rnn_type == 'LSTM':
            # LSTM keeps two hidden states
            h0 = (torch.zeros(h_shape), torch.zeros(h_shape))
        else:
            # RNN and GRU have one hidden state
            h0 = torch.zeros(h_shape)

        x, batch_sizes = torch._C._VariableFunctions._pack_padded_sequence(x, length.cpu().long(), batch_first=True)
        x = torch.nn.utils.rnn.PackedSequence(x, batch_sizes)

        h, _ = self.rnn(x, h0)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)

        #breakpoint()
        if torch.isnan(h).any():
            print("NaN Values in embedding vector!")
            breakpoint()
        
        return h

    def step(self, x, h):
        """Given input and hidden state produces the output and new state."""
        y, h = self.rnn(x, h)
        return y, h


class Hypernet(nn.Module):
    """Hypernetwork for incorporating conditional information.

    Args:
        config: Model configuration. See `dpp.model.ModelConfig`.
        hidden_sizes: Sizes of the hidden layers. [] corresponds to a linear layer.
        param_sizes: Sizes of the output parameters.
        activation: Activation function.
    """
    def __init__(self, config, hidden_sizes=[], param_sizes=[1, 1], activation=nn.Tanh()):
        super().__init__()
        self.history_size = config.history_size
        self.embedding_size = config.embedding_size
        self.activation = activation

        # Indices for unpacking parameters
        ends = torch.cumsum(torch.tensor(param_sizes), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        self.output_size = sum(param_sizes)
        layer_sizes = list(hidden_sizes) + [self.output_size]
        # Bias used in the first linear layer
        self.first_bias = nn.Parameter(torch.empty(layer_sizes[0]).uniform_(-0.1, 0.1))
        if config.use_history:
            self.linear_rnn = nn.Linear(self.history_size, layer_sizes[0], bias=False)
        if config.use_embedding:
            self.linear_emb = nn.Linear(self.embedding_size, layer_sizes[0], bias=False)
        # Remaining linear layers
        self.linear_layers = nn.ModuleList()
        for idx, size in enumerate(layer_sizes[:-1]):
            self.linear_layers.append(nn.Linear(size, layer_sizes[idx + 1]))

    def reset_parameters(self):
        self.first_bias.data.fill_(0.0)
        if hasattr(self, 'linear_rnn'):
            self.linear_rnn.reset_parameters()
            nn.init.orthogonal_(self.linear_rnn.weight)
        if hasattr(self, 'linear_emb'):
            self.linear_emb.reset_parameters()
            nn.init.orthogonal_(self.linear_emb.weight)
        for layer in self.linear_layers:
            layer.reset_parameters()
            nn.init.orthogonal_(linear.weight)

    def forward(self, h=None, emb=None):
        """Generate model parameters from the embeddings.

        Args:
            h: History embedding, shape (*, history_size)
            emb: Sequence embedding, shape (*, embedding_size)

        Returns:
            params: Tuple of model parameters.
        """
        # Generate the output based on the input
        if h is None and emb is None:
            # If no history or emb are provided, return bias of the final layer
            # 0.0 is added to create a new node in the computational graph
            # in case the output will be modified by an inplace operation later
            if len(self.linear_layers) == 0:
                hidden = self.first_bias + 0.0
            else:
                hidden = self.linear_layers[-1].bias + 0.0
        else:
            hidden = self.first_bias
            if h is not None:
                hidden = hidden + self.linear_rnn(h)
            if emb is not None:
                hidden = hidden + self.linear_emb(emb)
            for layer in self.linear_layers:
                hidden = layer(self.activation(hidden))

        # Partition the output
        if len(self.param_slices) == 1:
            return hidden
        else:
            return tuple([hidden[..., s] for s in self.param_slices])
