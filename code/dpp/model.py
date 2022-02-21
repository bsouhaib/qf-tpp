import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dpp

from dpp.utils import DotDict
from dpp.nn import BaseModule
import matplotlib.pyplot as plt


class Model(BaseModule):
    """Base model class.

    Attributes:
        rnn: RNN for encoding the event history.
        embedding: Retrieve static embedding for each sequence.
        decoder: Compute log-likelihood of the inter-event times given hist and emb.

    Args:
        config: General model configuration (see dpp.model.ModelConfig).
        decoder: Model for computing log probability of t given history and embeddings.
            (see dpp.decoders for a list of possible choices)
    """
    def __init__(self, config, decoder):
        super().__init__()
        self.use_history(config.use_history)
        self.use_embedding(config.use_embedding)
        self.use_marks(config.use_marks)

        self.rnn = dpp.nn.RNNLayer(config)
        if self.using_embedding:
            self.embedding = nn.Embedding(config.num_embeddings, config.embedding_size)
            self.embedding.weight.data.fill_(0.0)

        if self.using_marks:
            self.num_classes = config.num_classes
            self.mark_layer = nn.Sequential(
                nn.Linear(config.history_size, config.history_size),
                nn.ReLU(),
                nn.Linear(config.history_size, self.num_classes)
            )

        self.decoder = decoder

    def mark_nll(self, h, y):
        """Compute log likelihood and accuracy of predicted marks

        Args:
            h: History vector
            y: Out marks, true label

        Returns:
            loss: Negative log-likelihood for marks, shape (batch_size, seq_len)
            accuracy: Percentage of correctly classified marks
        """
        x = self.mark_layer(h)
        x = F.log_softmax(x, dim=-1)
        loss = F.nll_loss(x.view(-1, self.num_classes), y.view(-1), reduction='none').view_as(y)
        accuracy = (y == x.argmax(-1)).float()
        return loss, accuracy
    
    def expectation(self, input):
        if self.using_history:
            h = self.rnn(input) # has shape (batch_size, seq_len, rnn_hidden_size)
        else:
            h = None
        # Get sequence embedding
        if self.using_embedding:
            # has shape (batch_size, seq_len, embedding_size)
            emb = self.embedding(input.index).unsqueeze(1).repeat(1, input.out_time.shape[1], 1)
        else:
            emb = None

        return self.decoder.expectation(h, emb)


    def pi(self, input, N = None):
        if self.using_history:
            h = self.rnn(input) # has shape (batch_size, seq_len, rnn_hidden_size)
        else:
            h = None
        if self.using_embedding:
            emb = self.embedding(input.index).unsqueeze(1).repeat(1, input.out_time.shape[1], 1)
        else:
            emb = None
        
        t = input.out_time  # has shape (batch_size, seq_len)

        probs = torch.Tensor([0.05, 0.95])

        if N is not None:
            samples = self.decoder.sample(n_samples = N, h=h, emb=emb).detach()
            qf = torch.quantile(samples, probs, dim = 2).permute([1, 2, 0])
        else:
            params = self.decoder.get_params(h, emb)
            qf_1 = self.decoder.forward(torch.full(t.shape, probs[0]), params) 
            qf_2 = self.decoder.forward(torch.full(t.shape, probs[1]), params) 
        

    def score(self, input, scoring_rule = None, tau = None, return_details = True, N = None):
        """Compute log likelihood of the inter-event timesi in the batch.

        Args:
            input: Batch of data to score. See dpp.data.Input.

        Returns:
            time_log_prob: Log likelihood of each data point, shape (batch_size, seq_len)
            mark_nll: Negative log likelihood of marks, if using_marks is True
            accuracy: Accuracy of marks, if using_marks is True
        """
        # Encode the history with an RNN
        if self.using_history:
            h = self.rnn(input) # has shape (batch_size, seq_len, rnn_hidden_size)
        else:
            h = None
        # Get sequence embedding
        if self.using_embedding:
            # has shape (batch_size, seq_len, embedding_size)
            emb = self.embedding(input.index).unsqueeze(1).repeat(1, input.out_time.shape[1], 1)
        else:
            emb = None

 
        t = input.out_time  # has shape (batch_size, seq_len)

        if scoring_rule == "crps" or scoring_rule is None:
            results = self.decoder.crps(t, h, emb)

        elif scoring_rule == "crps_qapprox":
            results = self.decoder.crps_qapprox(t, h, emb, N = N) # 100 1000

        elif scoring_rule == "crps_qapprox_reg":
            results = self.decoder.crps_qapprox_reg(t, h, emb, training_mode = self.training, lengths = input.length, N = N) # 100 vs 1000
        
        elif scoring_rule == "crps_iapprox":
            results = self.decoder.crps_iapprox(t, h, emb, N = N)

        elif scoring_rule == "crps_sapprox":
            results =self.decoder.crps_sapprox(t, h, emb, N = N, return_details = return_details)

        elif scoring_rule == "logs":
            results = -self.decoder.log_prob(t, h, emb) # DO NOT FORGET THE MINUS HERE.

        elif scoring_rule == "quantile_score":
            results = self.decoder.quantile_score(tau, t, h, emb, params=None)

        elif scoring_rule == "calibration_error":
            results = self.decoder.calibration_error(t, h, emb, params=None, N =N)

        else:
            raise Exception("The scoring rule is not implemented.")

        return results

    def aggregate(self, values, lengths):
        """Calculate masked average of values.

        Sequences may have different lengths, so it's necessary to exclude
        the masked values in the padded sequence when computing the average.

        Arguments:
            values (list[tensor]): List of batches where each batch contains
                padded values, shape (batch size, sequence length)
            lengths (list[tensor]): List of batches where each batch contains
                lengths of sequences in a batch, shape (batch size)

        Returns:
            mean (float): Average value in values taking padding into account
        """

        if not isinstance(values, list):
            values = [values]
        if not isinstance(lengths, list):
            lengths = [lengths]

        total = 0.0
        for batch, length in zip(values, lengths):
            length = length.long()
            mask = torch.arange(batch.shape[1])[None, :] < length[:, None]
            mask = mask.float()

            if (torch.isnan(batch)).any():
                breakpoint()
                
            assert((~torch.isnan(batch)).all())
            # batch[torch.isnan(batch)] = 0 # set NaNs to 0 (from ifl-tpp)

            if len(batch.shape) == 2: 
                batch *= mask
                total += batch.sum()
            else:
                batch *= mask.unsqueeze(-1)
                total += torch.sum(batch, dim = [0, 1])

        total_length = sum([x.sum() for x in lengths])

        return total / total_length
    
    def get_total_calibration_loss(self, loader, N = None):
        loader_calib_error_mean = []
        loader_calib_error_max = []

        for input in loader:
            res_calib = self.score(input, "calibration_error", N = N)
            loader_calib_error_mean.append(res_calib[0])
            loader_calib_error_max.append(res_calib[1])

        return np.mean(loader_calib_error_mean), np.mean(loader_calib_error_max)

    def get_total_loss(self, loader, scoring_rule = None, tau = None, N = None):
        loader_score, loader_lengths = [], []
        for input in loader:
            loader_score.append(self.score(input, scoring_rule, tau, N = N).detach())
            loader_lengths.append(input.length.detach())
        return self.aggregate(loader_score, loader_lengths)
    
    def plot_functions(self, loader):
        input = next(iter(loader))
        if self.using_history:
            h = self.rnn(input) # has shape (batch_size, seq_len, rnn_hidden_size)
        else:
            h = None
        if self.using_embedding:
            # has shape (batch_size, seq_len, embedding_size)
            emb = self.embedding(input.index).unsqueeze(1).repeat(1, input.out_time.shape[1], 1)
        else:
            emb = None

        # sol = (cumwidths, widths, cumheights, heights, derivatives), normalized_params_tail, top
        params = self.decoder.get_params(h, emb)
        cumwidths = params[0][0][0, -1].detach()
        cumheights = params[0][2][0, -1].detach()
        derivatives = params[0][4][0, -1].detach()
        normalized_params_tail = params[1][0][0, -1].detach()
        top = params[2][0][-1, 0].detach()

        N = 1000 # 1000
        a = h.shape[0]
        b = h.shape[1]
        qtile = torch.empty(a, b, N)
        density = torch.empty(a, b, N)
        tau_all = torch.tensor(np.linspace(0, 1, N))
        for i in range(N):
            qtile[:, :, i] = self.decoder.forward(torch.full( (a, b), tau_all[i]), h = h, embed = emb)
            _, density[:, :, i] = self.decoder.inverse(qtile[:, :, i], h = h, embed = emb, compute_deriv = True)

        return tau_all, qtile, density, cumwidths, cumheights, normalized_params_tail, top, derivatives

    
    def get_total_loss_sampling(self, loader, return_details = True):
        loader_crps, loader_qs, loader_pits, loader_lengths, loader_means, loader_stds, loader_squared_errors, loader_abs_errors = [], [], [], [], [], [], [], []
        loader_target_interval_coverage, loader_abs_calibration_error, loader_mean_abs_calibration_error, loader_pi_coverage_prob =[], [], [], []
        for input in loader:
            results = self.score(input, "crps_sapprox", return_details = return_details)
            if return_details:
                crps_sampling, q_scores, pits, means, stds, squared_errors, abs_errors, target_interval_coverage, abs_calibration_error, mean_abs_calibration_error, pi_coverage_prob = results
            else:
                crps_sampling = results

            loader_crps.append(crps_sampling.detach())
            loader_lengths.append(input.length.detach())

            if return_details:
                loader_qs.append(q_scores.detach())
                # reduce size if N > 100
                loader_pits.append(pits.detach())

                loader_means.append(means.detach())
                loader_stds.append(stds.detach())

                loader_squared_errors.append(squared_errors.detach())
                loader_abs_errors.append(abs_errors.detach())

                # new
                loader_target_interval_coverage.append(target_interval_coverage.detach())
                loader_abs_calibration_error.append(abs_calibration_error.detach())
                loader_mean_abs_calibration_error.append(mean_abs_calibration_error)
                loader_pi_coverage_prob.append(pi_coverage_prob)


        
        crps = self.aggregate(loader_crps, loader_lengths)
        if return_details:
            quantile_scores = self.aggregate(loader_qs, loader_lengths)

            res_tuple = crps, quantile_scores, loader_pits, loader_means, loader_stds, loader_squared_errors, loader_abs_errors
            res_tuple = res_tuple + (loader_target_interval_coverage, loader_abs_calibration_error, loader_mean_abs_calibration_error, loader_pi_coverage_prob)
            return res_tuple

        else:
            return crps 
    
    def get_predictions_and_true_obs(self, loader, N = 500, taus = None):
        loader_results = []
        for input in loader:
            loader_results.append(self.get_all_quantities(input, N, taus))
        return loader_results

    def get_all_quantities(self, input, N, taus):
        ############
         # Encode the history with an RNN
        if self.using_history:
            h = self.rnn(input) # has shape (batch_size, seq_len, rnn_hidden_size)
        else:
            h = None
        # Get sequence embedding
        if self.using_embedding:
            # has shape (batch_size, seq_len, embedding_size)
            emb = self.embedding(input.index).unsqueeze(1).repeat(1, input.out_time.shape[1], 1)
        else:
            emb = None
        ############
        
        t = input.out_time  # has shape (batch_size, seq_len)

        samples = self.decoder.sample(N, h, emb).detach()

        quantiles = None
        q_densities = None
        params = None
        if taus is not None:
            params = self.decoder.get_params(h = h, embed = emb)
            ###
            all_q, all_d = [], []
            for tau in taus:
                q = self.decoder.forward(torch.full(t.shape, tau), params = params)
                _, d = self.decoder.inverse(q, params = params, compute_deriv = True)

                all_q.append(q)
                all_d.append(d)

            quantiles = torch.stack(all_q, dim = 2)
            q_densities = torch.stack(all_d, dim = 2)

        return (t, samples, quantiles, q_densities, params) 


    def get_samples(self, input, N):
        # Encode the history with an RNN
        if self.using_history:
            h = self.rnn(input) # has shape (batch_size, seq_len, rnn_hidden_size)
        else:
            h = None
        # Get sequence embedding
        if self.using_embedding:
            # has shape (batch_size, seq_len, embedding_size)
            emb = self.embedding(input.index).unsqueeze(1).repeat(1, input.out_time.shape[1], 1)
        else:
            emb = None
 
        #t = input.out_time  # has shape (batch_size, seq_len)
        samples = self.decoder.sample(N, h, emb).detach()
        #return (t, samples)
        return samples
    
    def get_quantiles(self, input, taus):
        # Encode the history with an RNN
        if self.using_history:
            h = self.rnn(input) # has shape (batch_size, seq_len, rnn_hidden_size)
        else:
            h = None
        # Get sequence embedding
        if self.using_embedding:
            # has shape (batch_size, seq_len, embedding_size)
            emb = self.embedding(input.index).unsqueeze(1).repeat(1, input.out_time.shape[1], 1)
        else:
            emb = None
 
        t = input.out_time  # has shape (batch_size, seq_len)
        
        params = self.decoder.get_params(h = h, embed = emb)
        all_q = []
        for tau in taus:
            all_q.append(self.decoder.forward(torch.full(t.shape, tau), params = params).detach())

        #return (t, torch.stack(all_q, dim = 2))
        return torch.stack(all_q, dim = 2)

class ModelConfig(DotDict):
    """Configuration of the model.

    This config only contains parameters that need to be know by all the
    submodules. Submodule-specific parameters are passed to the respective
    constructors.

    Args:
        use_history: Should the model use the history embedding?
        history_size: Dimension of the history embedding.
        rnn_type: {'RNN', 'LSTM', 'GRU'}: RNN architecture to use.
        use_embedding: Should the model use the sequence embedding?
        embedding_size: Dimension of the sequence embedding.
        num_embeddings: Number of unique sequences in the dataset.
        use_marks: Should the model use the marks?
        mark_embedding_size: Dimension of the mark embedding.
        num_classes: Number of unique mark types, used as dimension of output
    """
    def __init__(self,
                 use_history=True,
                 history_size=32,
                 rnn_type='RNN',
                 use_embedding=False,
                 embedding_size=32,
                 num_embeddings=None,
                 use_marks=False,
                 mark_embedding_size=64,
                 num_classes=None):
        super().__init__()
        # RNN parameters
        self.use_history = use_history
        self.history_size = history_size
        self.rnn_type = rnn_type

        # Sequence embedding parameters
        self.use_embedding = use_embedding
        self.embedding_size = embedding_size
        if use_embedding and num_embeddings is None:
            raise ValueError("Number of embeddings has to be specified")
        self.num_embeddings = num_embeddings

        self.use_marks = use_marks
        self.mark_embedding_size = mark_embedding_size
        self.num_classes = num_classes
