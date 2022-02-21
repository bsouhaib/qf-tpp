import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from torch.distributions import constraints

from dpp.utils import searchsorted
from dpp.centers import splines
from dpp.nn import BaseModule, Hypernet
from dpp.utils import clamp_preserve_gradients

from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from datetime import datetime

from matplotlib.backends.backend_pdf import PdfPages



class QuantileFunction(BaseModule):
	def __init__(self, config):
		super().__init__()
		self.use_history(config.use_history)
		self.use_embedding(config.use_embedding)
	
	def invcdf(self, tau, h=None, emb=None, params = None):
		return self.forward(tau, h = h, embed = emb, params = params) 

	
	def crps_iapprox(self, y, h, embed, N = 100):
		params = self.get_params(h, embed)
		all_tau_bar = self.inverse(y, params, return_param = False)

		tau_all = torch.tensor(np.linspace(0, 1-1e-9, N)) 

		a = y.shape[0]
		b = y.shape[1]
		qtile = torch.empty(a, b, N)
		tau_qtau = torch.empty(a, b, N)
		for i in range(N):
			qtile[:, :, i] = self.forward( torch.full(y.shape, tau_all[i]) , params)
			tau_qtau[:, :, i] = qtile[:, :, i] * tau_all[i]
		
		part_2 = torch.trapz(tau_qtau, tau_all) 
		part_3 =  torch.empty(a, b)
		part1_2_1 =  torch.empty(a, b)
		
		vec = torch.linspace(0, 1, N) 
		for i in range(all_tau_bar.shape[0]):
			for j in range(all_tau_bar.shape[1]):
				tau_bar = all_tau_bar[i, j]

				idx = torch.where(tau_all >= tau_bar)[0]
				my_tau_s = vec * (1-1e-9 - tau_bar) + tau_bar
				part1_2_1[i, j] = torch.trapz(torch.tensor(1.0).repeat_interleave(len(my_tau_s)), my_tau_s)
				part_3[i, j] = torch.trapz(qtile[i, j, idx], tau_all[idx])

		# part1
		part1_1 = y * torch.trapz(tau_all, tau_all) 
		part1_2 = y * part1_2_1
		part_1 = part1_1 - part1_2

		#print(part_1[5, 9])
		#print(part_2[5, 9])
		#print(part_3[5, 9])

		return 2 * (part_1 - part_2 + part_3)
	
	def crps_qapprox(self, y, h, embed, N = 100):

		params = self.get_params(h, embed)

		tau_all = torch.tensor(np.linspace(0, 1-1e-9, N)) # more precise
		qs = torch.empty(y.shape[0], y.shape[1], N)
		for i in range(N):
			qs[:, :, i] = self.quantile_score(tau_all[i], y, params = params)
		

		return torch.trapz(qs, tau_all)


	def crps_qapprox_reg(self, y, h, embed, training_mode, lengths, N = 100):
	
		params = self.get_params(h, embed)

		tau_all = torch.tensor(np.linspace(0, 1-1e-9, N)) # more precise
		qf = torch.empty(y.shape[0], y.shape[1], N)
		qs = torch.empty(y.shape[0], y.shape[1], N)
		for i in range(N):
			qf[:, :, i] = self.forward(torch.full(y.shape, tau_all[i]), params)
			qs[:, :, i] = (((y < qf[:, :, i]).int() - tau_all[i]) * (qf[:, :, i] - y)) * 2.0

		loss = torch.trapz(qs, tau_all)
		
		if training_mode and  self.calib_regularization != 0:
			total_length = sum([x.sum() for x in lengths])
			mask = torch.arange(y.shape[1])[None, :] < lengths.long()[:, None]
			
			# Non-differentiable
			#res = torch.abs(tau_all  - torch.sum(y[mask].unsqueeze(-1) < qf[mask, :], dim = 0) / y[mask].shape[0])
			
			# Differentiable
			#indicator = (norm > threshold).float() - torch.sigmoid(norm - threshold).detach() + torch.sigmoid(norm - threshold)
			indicator = (qf[mask, :] > y[mask].unsqueeze(-1)).float() - torch.sigmoid(qf[mask, :] - y[mask].unsqueeze(-1)).detach() + torch.sigmoid(qf[mask, :]  - y[mask].unsqueeze(-1))
			
			# ABSOLUTE VALUE
			res = torch.abs(tau_all  -  (torch.sum(indicator, dim = 0)/y[mask].shape[0]))

			reg = torch.mean(res) * total_length # IT WILL BE DIVIDED BY TOTAL_LENGTH IN AGGREGATE

			lam = self.calib_regularization

			return loss + lam * reg 
		else:
			return loss

			

	def calibration_error(self, y, h = None, embed = None, params = None, N = 100):
		if params is None:
			params = self.get_params(h, embed)
		
		tau_all = torch.tensor(np.linspace(0, 1-1e-9, N)) # more precise
		qf = torch.empty(y.shape[0], y.shape[1], N)
		for i in range(N):
			qf[:, :, i] = self.forward(torch.full(y.shape, tau_all[i]), params)

		res = torch.abs(tau_all - torch.sum(y.unsqueeze(-1) <= qf, dim = 1)/y.shape[1])
		return torch.mean(res).item(), torch.max(res).item()
    
	def quantile_score(self, tau, y, h = None, embed = None, params = None):

		if params is None:
			params = self.get_params(h, embed)
		
		all_q = self.forward(torch.full(y.shape, tau), params)
		return (((y < all_q).int() - tau) * (all_q - y)) * 2.0

	def get_params(self, h, embed):
		raise NotImplementedError

	def forward(self, inputs, params = None, h = None, embed = None, return_param = False, compute_deriv = False, compute_sderiv = False): 
		raise NotImplementedError

	def inverse(self, inputs, params = None, h = None, embed = None, return_param = False, compute_deriv = False, compute_sderiv = False):
		raise NotImplementedError

	def crps(self, y, h, embed): 
		raise NotImplementedError

class ExponentialQuantileFunction(QuantileFunction):
	def __init__(self, config, hypernet_hidden_sizes):
		super().__init__(config)
		self.hypernet = Hypernet(config, hidden_sizes=hypernet_hidden_sizes, param_sizes= [1]) 
		self.reset_parameters()
		self.epsilon = 1e-9

	def reset_parameters(self):
		self.hypernet.reset_parameters()

	def get_params(self, h, embed):
		if not self.using_history:
			h = None
		if not self.using_embedding:
			embed = None
		unnormalized_lambda = self.hypernet(h, embed)
		lambda_rate = F.softplus(unnormalized_lambda).squeeze(-1)
		return lambda_rate

	def forward(self, inputs, params = None, h = None, embed = None, return_param = False, compute_deriv = False, compute_sderiv = False): 
		if params is None:
			params = self.get_params(h, embed)
		# Quantile function of exponential distribution
		return -torch.log((1.0 - inputs) + self.epsilon)/params

	def inverse(self, inputs, params = None, h = None, embed = None, return_param = False, compute_deriv = False, compute_sderiv = False):
		if params is None:
			params = self.get_params(h, embed)
		# CDF of exponential distribution
		return 1.0 - torch.exp(-params * inputs) + self.epsilon
	
	def log_prob(self, y, h, embed, training = False, reg_lambda = 0):
		lam = self.get_params(h, embed)
		log_p = lam.log() - lam * y
		return log_p

	def crps(self, y, h, embed, training = False, reg_lambda = 0): 
		lambda_rate = self.get_params(h, embed)
		return y - 2 * (self.inverse(y, params = lambda_rate)/lambda_rate) + 1.0/(2.0 * lambda_rate)

class PieceWiseQuantileFunction(QuantileFunction):
	def __init__(self, config, center_model, tail_model, hypernet_hidden_sizes, train_widths, log_and_scaling, scale_init, calib_regularization):
		super().__init__(config)
		
		self.center_model = center_model
		self.tail_model = tail_model
		
		self.train_widths = train_widths
		self.log_and_scaling = log_and_scaling
		self.scale_init = scale_init

		self.calib_regularization = calib_regularization

				
		if not self.train_widths:
		 	param_sizes_center = [self.center_model.num_bins, self.center_model.num_bins+1]
		else:
			param_sizes_center = [self.center_model.num_bins, self.center_model.num_bins+1, self.center_model.num_bins]

		param_sizes_top = [1] 
		param_sizes = param_sizes_center + param_sizes_top

		self.n_params_center = len(param_sizes_center)
		self.n_params_top = len(param_sizes_top)

		self.hypernet = Hypernet(config, hidden_sizes=hypernet_hidden_sizes, param_sizes= param_sizes) 


	def _transform(self, inputs, params, return_param, compute_deriv, compute_sderiv, inverse = False):

		if compute_sderiv:
			raise Exception("CARREFUL. Sderiv is not valid with the scaling!")
		
		if inverse and self.log_and_scaling:
			inputs = torch.log(1.0 + inputs)/self.scale_init


		if inverse:
			inside_interval_mask = (inputs < params[-1].squeeze())
		else:
			inside_interval_mask = (inputs < self.center_model.right)

		outside_interval_mask = ~inside_interval_mask
		
		outputs = torch.zeros_like(inputs)
		if compute_deriv:
			deriv = torch.zeros_like(inputs)
		if compute_sderiv:
			sderiv = torch.zeros_like(inputs)

		if (torch.any(outside_interval_mask).item()):
			params_tail = [mat[outside_interval_mask] for mat in params[1]]
			params_top = params[2][outside_interval_mask].flatten()

			res = self.tail_model.upper_tail(inputs[outside_interval_mask], *params_tail, params_top, inverse, compute_deriv, compute_sderiv)

			outputs[outside_interval_mask] = res[0] 
			if compute_deriv:
				deriv[outside_interval_mask] = res[1]
			if compute_sderiv:
				sderiv[outside_interval_mask] = res[2]


		if (torch.any(inside_interval_mask).item()):
			params_center = [mat[inside_interval_mask] for mat in params[0]]
			res = self.center_model.transform(inputs[inside_interval_mask], *params_center, inverse, compute_deriv, compute_sderiv)
			outputs[inside_interval_mask], bin_idx, input_cumwidths, input_bin_widths = res[:4]
			if compute_deriv:
				deriv[inside_interval_mask] = res[4]
			if compute_sderiv:
				sderiv[inside_interval_mask]  = res[5]

		#
		if self.log_and_scaling and not inverse:
			outputs = torch.exp(outputs * self.scale_init) - 1.0


		sol = outputs
		if return_param:
			if torch.all(outside_interval_mask).item():
				print("All outside!")
				breakpoint()

			sol = (sol, bin_idx, inside_interval_mask, input_cumwidths, input_bin_widths)

		if compute_deriv:
			if self.log_and_scaling:
				if inverse:
					deriv = deriv / (torch.exp(inputs * self.scale_init)  * self.scale_init)
				else:
					deriv = deriv * (outputs + 1.0) * self.scale_init
				
			sol = (sol, deriv)
		if compute_sderiv:
			sol = (sol, sderiv)
				
		return sol
		


	def forward(self, inputs, params = None, h = None, embed = None, return_param = False, compute_deriv = False, compute_sderiv = False): 
		if params is None:
			params = self.get_params(h, embed)
		return self._transform(inputs, params, return_param, compute_deriv, compute_sderiv, inverse = False)
		
	
	def inverse(self, inputs, params = None, h = None, embed = None, return_param = False, compute_deriv = False, compute_sderiv = False):
		if params is None:
			params = self.get_params(h, embed)
		return self._transform(inputs, params, return_param, compute_deriv, compute_sderiv, inverse = True)
	
	def log_prob(self, y, h = None, embed = None, training = False, reg_lambda = 0):
		if False:
			self.plot_details(y, h, embed)

		params = self.get_params(h, embed)
		_, fderiv = self.inverse(y, params = params, compute_deriv = True, compute_sderiv = False)

		log_prob = torch.log(fderiv + 1e-9) # added 1e-9

		return log_prob
	
	def get_params(self, h, embed):
		unnormalized_params = self.hypernet(h, embed)
		if not self.train_widths:
			unnormalized_heights, unnormalized_derivatives = unnormalized_params[:self.n_params_center]
			unnormalized_widths = torch.log(torch.zeros_like(unnormalized_heights) + 1.0/unnormalized_heights.shape[2])
		else:
			unnormalized_widths, unnormalized_derivatives, unnormalized_heights = unnormalized_params[:self.n_params_center]
		
		unnormalized_top = unnormalized_params[-1]

		# NORMALIZED TOP
		normalized_top = torch.sigmoid(unnormalized_top) * (self.center_model.max_qtail  - self.center_model.min_qtail) + self.center_model.min_qtail

		# NORMALIZED DERIVATIVES
		min_derivative = self.center_model.min_derivative
		normalized_derivatives = min_derivative + F.softplus(unnormalized_derivatives)
		
		normalized_params_tail = self.tail_model.get_normalized_lambda(normalized_derivatives[:,:, -1:])		

		cumwidths, widths = splines.normalize_params(unnormalized_widths, 
														self.center_model.min_bin_width, 
														self.center_model.num_bins, 
														self.center_model.left, 
														self.center_model.right)
		cumheights, heights = splines.normalize_params(unnormalized_heights, 
														self.center_model.min_bin_height, 
														self.center_model.num_bins,
														self.center_model.bottom, 
														normalized_top)
		
		return (cumwidths, widths, cumheights, heights, normalized_derivatives), (normalized_params_tail,), normalized_top



	def crps(self, y, h, embed): 

		if self.log_and_scaling:
			raise("Closed-form CRPS with transformation is not yet implemented!")

		params = self.get_params(h, embed)

		params_center_list = params[:-2][0]
		params_lambda = params[-2][0].squeeze(-1)
		params_top = params[-1].squeeze(-1)

		all_tau_bar, bin_idx, inside_interval_mask, input_cumwidths, input_bin_widths = self.inverse(y, params, return_param = True)

		part1 = y * (all_tau_bar - 0.5)
		part2_1, part3_1 = self.center_model.crps(y, all_tau_bar, inside_interval_mask, bin_idx, *params_center_list, input_cumwidths, input_bin_widths)		
		part2_2, part3_2 = self.tail_model.crps(all_tau_bar, ~inside_interval_mask, params_lambda, params_top)

		assert((part2_1 >= 0).all() and (part3_1 >= 0).all() and (part2_2 >= 0).all()  and (part3_2 >= 0).all())  

		crps = 2.0 * (part1 - (part2_1 + part2_2) + (part3_1 + part3_2))
			
		if not (crps > 0).all():
			print("There are negative values!")
			breakpoint()

		return crps

	def expectation(self, h, embed):

		if self.log_and_scaling:
			raise("Closed-form expectation with transformation is not implemented!")

		params = self.get_params(h, embed)

		params_center_list = params[:-2][0]
		params_lambda = params[-2][0].squeeze(-1)
		params_top = params[-1].squeeze(-1)

		expectation_part1 = self.center_model.integral_qf(*params_center_list)		
		expectation_part2 = self.tail_model.integral_qf(params_lambda, params_top)

		return expectation_part1 + expectation_part2

