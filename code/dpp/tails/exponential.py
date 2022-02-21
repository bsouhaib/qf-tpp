import torch
import numpy as np
import torch.nn.functional as F


class ExponentialTail:
    def __init__(self, right):
        self.right = right
    
    def upper_tail(self, inputs, params_tail, params_top, inverse = False, compute_deriv = False, compute_sderiv = False):
        lambda_param = params_tail.squeeze() 
        if inverse: # CDF

            outputs = 1.0 - ( torch.exp(-lambda_param * (inputs - params_top)) * (1.0 - self.right))
            outputs[(outputs==1)] = torch.tensor(np.nextafter(1, 0, dtype=np.dtype('float32')))
            if (outputs >= 1).any():
                breakpoint()
                assert((outputs < 1).all())

            if compute_deriv: # PDF
                deriv =  lambda_param * ( torch.exp(-lambda_param * (inputs - params_top)) * (1.0 - self.right))
                
        else: # INVCDF
            if torch.any(inputs == 1):
                inputs[(inputs==1)] = torch.tensor(np.nextafter(1, 0, dtype=np.dtype('float32')))

            outputs = params_top - (1.0/lambda_param) * torch.log((1.0 - inputs)/(1.0 - self.right))

            
            assert(not torch.any(torch.isinf(outputs)))

            if compute_deriv:
                deriv =  1.0/(lambda_param * (1.0 - inputs))
            if compute_sderiv:
                sderiv = 1.0/(lambda_param * (1.0 - inputs).pow(2))
    

        if compute_deriv and not compute_sderiv:
            return outputs, deriv, None
        elif compute_deriv and compute_sderiv and not inverse:
            return outputs, deriv, sderiv
        else:
            return outputs, None, None

    def upper_tail_derivatives(self, params_tail):
        lambda_param = params_tail[0]
        return (1.0/lambda_param) * (1.0/(1.0 - self.right))

    def integral_tau_qtail_right_to_one(self, lambda_param, top_param):
        return -((self.right - 1.0) * ((2.0 * self.right + 2.0) * top_param * lambda_param + self.right + 3.0))/(4.0 * lambda_param)
        
    def integral_qtail_lower_to_one(self, lower, lambda_param, top_param):
        assert(torch.min(lower) >= self.right and torch.max(lower) < 1) # IF lower is equal to one, then problem

        return ((1 - lower) * (top_param * lambda_param  + 1 - torch.log((1 - lower) / (1 - self.right))))/lambda_param
    
    def crps(self, all_tau_bar, outside_interval_mask, params_tail, top_param):
        lambda_param = params_tail[0].squeeze()
        #### PART 2_2 #####
        part2_2 = self.integral_tau_qtail_right_to_one(lambda_param, top_param)

        #### PART 3_2 #####
        lower = torch.zeros_like(all_tau_bar) + self.right
        
        if torch.any(outside_interval_mask).item():
            lower[outside_interval_mask] = all_tau_bar[outside_interval_mask]
        
        part3_2 = self.integral_qtail_lower_to_one(lower, lambda_param, top_param)
        
        return part2_2, part3_2
        
    def get_param_sizes(self):
        return [1]

    def normalize_allparams(self, unnormalized_params):
        return (1e-5 + F.softplus(unnormalized_params[0]), ) # we divide by lambda above -> instability with small values

    def get_normalized_lambda(self, normalized_derivatives_tail):
        return (1.0/normalized_derivatives_tail) * (1.0/(1.0 - self.right))

    def integral_qf(self, lambda_param, top_param):
        return self.integral_qtail_lower_to_one(torch.Tensor([self.right]), lambda_param, top_param)

