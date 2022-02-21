import torch
from torch.nn import functional as F
from dpp.utils import searchsorted
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datetime import datetime

from dpp.utils import clamp_preserve_gradients


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""
    pass

def normalize_params(unnormalized_params, min_bin_param, num_bins, lower_bound, upper_bound):
    params = F.softmax(unnormalized_params, dim = -1)
    
    params = min_bin_param + (1 - min_bin_param * num_bins) * params
    cumparams = torch.cumsum(params, dim = -1)
    cumparams = F.pad(cumparams, pad=(1, 0), mode='constant', value=0.0)

    cumparams = (upper_bound - lower_bound) * cumparams + lower_bound
    
    cumparams[..., 0] = lower_bound
    
    #cumparams[..., -1] = upper_bound
    if torch.is_tensor(upper_bound):
        cumparams[..., -1] = upper_bound.squeeze()
    else:
        cumparams[..., -1] = upper_bound

    params = cumparams[..., 1:] - cumparams[..., :-1]
    return cumparams, params;


def normalize_all_params(unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
num_bins, min_bin_width, min_bin_height, min_derivative, left, right, bottom, top):
    cumwidths, widths = normalize_params(unnormalized_widths, min_bin_width, num_bins, left, right)
    cumheights, heights = normalize_params(unnormalized_heights, min_bin_height, num_bins, bottom, top)

    derivatives = min_derivative + F.leaky_relu(unnormalized_derivatives, negative_slope=0.01)
    derivatives[(derivatives < 0)] = min_derivative 

    return cumwidths, widths, cumheights, heights, derivatives

def get_constant(widths, cumheights, heights, derivatives): # cumwidths not used
    delta = heights / widths
    derivatives_plus_one = derivatives[..., 1:]

    c_1 = cumheights[..., 0:-1]
    c_2 = heights * (delta - derivatives[..., 0:-1])
    c_3 = heights * derivatives[..., 0:-1]
    
    c_4_raw = derivatives_plus_one + derivatives[..., 0:-1] - 2 * delta
    c_4 =  F.threshold(c_4_raw, 1e-4, 0.0) - F.threshold(-c_4_raw,  1e-4, 0.0) 
    
    c_5 = delta
    return c_1, c_2, c_3, c_4, c_5

def rational_quadratic_spline(inputs,num_bins, 
                              cumwidths, 
                              widths, 
                              cumheights, 
                              heights, 
                              derivatives,
                              inverse=False,
                              compute_deriv = False,
                              compute_sderiv = False,
                              min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                              min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                              min_derivative=DEFAULT_MIN_DERIVATIVE):
    

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]
    
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]

    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]


    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives
                                             + input_derivatives_plus_one
                                             - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives
                                              + input_derivatives_plus_one
                                              - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        
        assert (discriminant >= 0).all()

        #root = (2 * c) / (-b - torch.sqrt(discriminant))
        # Numerical approximations can make root > 1 or root < 0   
        root = torch.clamp((2 * c) / (-b - torch.sqrt(discriminant)), 0, 1)

        outputs = root * input_bin_widths + input_cumwidths

        if compute_deriv or compute_sderiv:
            root_one_minus_root = root * (1 - root)
            denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * root_one_minus_root)
            derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                        + 2 * input_delta * root_one_minus_root
                                                        + input_derivatives * (1 - root).pow(2))
            #deriv = derivative_numerator/denominator.pow(2)
            deriv = denominator.pow(2)/derivative_numerator     


        if compute_sderiv:
            # evalute f_deriv at f_inverse and compute 1/result
            Z = (input_derivatives_plus_one + input_derivatives - 2 * input_delta)
            term = 2.0 * input_delta.pow(2) * (root * Z + (input_delta - input_derivatives)) * denominator.pow(2) \
                - 2.0 * derivative_numerator * denominator * Z * (1.0 - 2.0 * root)
            second_deriv = - (term * denominator.pow(2))/derivative_numerator.pow(3)

    else:

        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2.0 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        if compute_deriv or compute_sderiv:
            derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                     + 2.0 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - theta).pow(2))
            deriv = derivative_numerator/denominator.pow(2)
    
        if compute_sderiv:
            c_4 = (input_derivatives_plus_one + input_derivatives - 2 * input_delta)
            c_2 = (input_delta - input_derivatives)
            term = 2.0 * input_delta.pow(2) * (theta * c_4 + c_2) * denominator.pow(2) - 2.0 * derivative_numerator * denominator * c_4 * (1.0 - 2.0 * theta)
            second_deriv = term / denominator.pow(4)
    

    if compute_deriv and not compute_sderiv: # 1 0
        return outputs, bin_idx, input_cumwidths, input_bin_widths, deriv
    elif compute_deriv and compute_sderiv: # 1 1
        return outputs, bin_idx, input_cumwidths, input_bin_widths, deriv, second_deriv
    else:
        return outputs, bin_idx, input_cumwidths, input_bin_widths

def integral_rational(A, B, a, b, c, x_l = 0, x_u = 1.0):
    # Integral of A x + B / a x^2 + b x + c in the interval [x_l, x_u]

    delta = b**2 - 4.0 * a * c
    
    res = torch.zeros_like(delta)

    if torch.any(delta > 0).item():
        idx_pos = torch.where(delta > 0)

        x_1 = (-b[idx_pos] - torch.sqrt(delta[idx_pos]))/(2*a[idx_pos])
        x_2 = (-b[idx_pos] + torch.sqrt(delta[idx_pos]))/(2*a[idx_pos])

        R = (A[idx_pos] * x_1 + B[idx_pos]) / (x_1 - x_2)
        L = (A[idx_pos] * x_2 + B[idx_pos]) / (x_2 - x_1)

        part_one = R * ( torch.log(torch.abs(x_u[idx_pos] - x_1)) - torch.log(torch.abs(x_l[idx_pos] - x_1)) )
        part_two = L * ( torch.log(torch.abs(x_u[idx_pos] - x_2)) - torch.log(torch.abs(x_l[idx_pos] - x_2)) )
        
        res[idx_pos] = ((part_one + part_two)/a[idx_pos])
    
    if torch.any(delta < 0).item():
        idx_neg = torch.where(delta < 0)

        t_l = x_l[idx_neg] + (b[idx_neg]/(2.0 * a[idx_neg]))
        t_u = x_u[idx_neg] + (b[idx_neg]/(2.0 * a[idx_neg]))

        m_squared = (4.0 * a[idx_neg] * c[idx_neg] - b[idx_neg]**2)/(4.0 * a[idx_neg]**2) # alwyas positive sinice b^2 - 4ac < 0
        part_one = A[idx_neg] * 0.5 * ( torch.log(t_u **2 + m_squared) - torch.log(t_l**2 + m_squared) )

        B_prime = B[idx_neg] - ((A[idx_neg] * b[idx_neg])/(2.0 * a[idx_neg]))
        m = torch.sqrt(m_squared) 
        part_two = (B_prime/m) * ( torch.atan(t_u/m) - torch.atan(t_l/m) )

        res[idx_neg] = ((part_one + part_two)/a[idx_neg])

    if torch.any(delta == 0).item():
        idx_null = torch.where(delta == 0)

        x_0 = -b[idx_null]/(2*a[idx_null])
        R = A[idx_null]
        L = B[idx_null] + A[idx_null] * x_0

        part_one = R * ( torch.log(torch.abs(x_u[idx_null] - x_0)) - torch.log(torch.abs(x_l[idx_null] - x_0)) )
        part_two = -L * ( (1.0/(x_u[idx_null] - x_0)) - (1.0/(x_l[idx_null] - x_0)) )
        res[idx_null] = ((part_one + part_two)/a[idx_null])

    return res


def integral_qf(cumwidths,  widths, cumheights, heights, derivatives): 

    num_bins = widths.shape[-1]
    c_1, c_2, c_3, c_4, c_5 = get_constant(widths, cumheights, heights, derivatives)

    tensor_shape = c_1.shape
    _c_1 = c_1.reshape(-1, tensor_shape[-1])
    _c_2 = c_2.reshape(-1, tensor_shape[-1])
    _c_3 = c_3.reshape(-1, tensor_shape[-1])
    _c_4 = c_4.reshape(-1, tensor_shape[-1])
    _c_5 = c_5.reshape(-1, tensor_shape[-1])
    n_var = _c_1.shape[0]

    res_allk = torch.zeros_like(_c_1)
    lower_a = torch.zeros(n_var).unsqueeze(1)
    upper_b = torch.zeros(n_var).unsqueeze(1) + 1
    
    for k in range(num_bins):
        bin_idx_k = torch.full((n_var, 1), k)
        res_allk[:, k:k+1] = integral_rk(lower_a, upper_b, bin_idx_k, _c_1, _c_2, _c_3, _c_4, _c_5) 

    res = torch.sum(widths.squeeze() * res_allk, -1).unsqueeze(0)

    return res

def crps(y, all_tau_bar, inside_interval_mask, bin_idx, cumwidths, 
            widths, cumheights, heights, derivatives, input_cumwidths, input_bin_widths):

    num_bins = widths.shape[-1]

    c_1, c_2, c_3, c_4, c_5 = get_constant(widths, cumheights, heights, derivatives)

    tensor_shape = c_1.shape
    _c_1 = c_1.reshape(-1, tensor_shape[-1])
    _c_2 = c_2.reshape(-1, tensor_shape[-1])
    _c_3 = c_3.reshape(-1, tensor_shape[-1])
    _c_4 = c_4.reshape(-1, tensor_shape[-1])
    _c_5 = c_5.reshape(-1, tensor_shape[-1])
    n_var = _c_1.shape[0]

    #### PART 2.1 #####
    res1_allk = torch.zeros_like(_c_1)
    res2_allk = torch.zeros_like(_c_1)
    lower_a = torch.zeros(n_var).unsqueeze(1)
    upper_b = torch.zeros(n_var).unsqueeze(1) + 1
    
    for k in range(num_bins):
        bin_idx_k = torch.full((n_var, 1), k)
        res1_allk[:, k:k+1] = integral_xi_rk(lower_a, upper_b, bin_idx_k, _c_1, _c_2, _c_3, _c_4, _c_5) 
        res2_allk[:, k:k+1] = integral_rk(lower_a, upper_b, bin_idx_k, _c_1, _c_2, _c_3, _c_4, _c_5) 

    part2_1 = torch.sum(res1_allk.reshape(tensor_shape) * widths **2, dim = -1) + \
         torch.sum(res2_allk.reshape(tensor_shape) * cumwidths[..., 0:-1] * widths, dim = -1)
 
    #### PART 3.1 #####
    res1 = torch.zeros_like(y)
    res2 = torch.zeros_like(y)
    if(torch.any(inside_interval_mask).item()):
 
        n = torch.sum(inside_interval_mask)
        
        lower_a = ((all_tau_bar[inside_interval_mask] - input_cumwidths)/input_bin_widths).unsqueeze(1)
        upper_b = torch.ones(n).unsqueeze(1)

        ############
        if (lower_a == upper_b).any():
            print("Lower_a EQUALS upper_b!")

        if (lower_a > upper_b).any():
            id = torch.where(lower_a > upper_b)
            #lower_a[id] = upper_b[id]
            print("------")
            print("Lower_a > upper_b !")
            print((upper_b[id] - lower_a[id]).max())
            print("------")
            breakpoint()
        ############


        c_1_inside = c_1[inside_interval_mask, ...]
        c_2_inside = c_2[inside_interval_mask, ...]
        c_3_inside = c_3[inside_interval_mask, ...]
        c_4_inside = c_4[inside_interval_mask, ...]
        c_5_inside = c_5[inside_interval_mask, ...]

        _widths = widths[inside_interval_mask]
        res1[inside_interval_mask] = _widths.gather(-1, bin_idx)[..., 0] * integral_rk(lower_a, upper_b, bin_idx, c_1_inside, c_2_inside, c_3_inside, c_4_inside, c_5_inside)[..., 0]
        
        ids = torch.where(inside_interval_mask.flatten())[0]
        #  res2[inside_interval_mask] = torch.stack([torch.dot(res2_allk[pos, range(bin_idx[i] + 1,num_bins)], _widths.T[range(bin_idx[i] + 1, num_bins), i]) if bin_idx[i] < num_bins - 1 else torch.tensor(0.0) for i, pos in enumerate(ids) ])

        res2[inside_interval_mask] = torch.stack([torch.dot(res2_allk[pos, bin_idx[i] + 1:], \
            _widths[i, bin_idx[i] + 1:]) if bin_idx[i] < num_bins - 1 else torch.tensor(0.0) for i, pos in enumerate(ids) ])
 
 

    part3_1 = res1 + res2
    
    if torch.any(res1 < 0):
        breakpoint()
    
    if torch.any(res2 < 0):
        breakpoint()


    return part2_1, part3_1

def integral_rk(lower, upper, bin_idx, c_1, c_2, c_3, c_4, c_5):
    
    c_1_vec = c_1.gather(-1, bin_idx);
    c_2_vec = c_2.gather(-1, bin_idx); c_3_vec = c_3.gather(-1, bin_idx); 
    c_4_vec = c_4.gather(-1, bin_idx); c_5_vec = c_5.gather(-1, bin_idx);

    res = torch.zeros_like(c_1_vec)
    id_zero = (c_4_vec == 0)
    id_nzero = ~id_zero 
    if any(id_zero):
        _c_25_vec = c_2_vec[id_zero, None]/c_5_vec[id_zero, None]
        _c_35_vec = c_3_vec[id_zero, None]/c_5_vec[id_zero, None]
        _c_1_vec = c_1_vec[id_zero, None]
        _lower = lower[id_zero, None]
        _upper = upper[id_zero, None]
        res[id_zero, None] = (_c_25_vec/3.0) * (_upper.pow(3) - _lower.pow(3)) + (_c_35_vec/2.0) * (_upper.pow(2) - _lower.pow(2)) + _c_1_vec * (_upper - _lower) 
        
    if any(id_nzero):
        _c_1_vec = c_1_vec[id_nzero, None]
        _c_2_vec = c_2_vec[id_nzero, None]
        _c_3_vec = c_3_vec[id_nzero, None]
        _c_4_vec = c_4_vec[id_nzero, None]
        _c_5_vec = c_5_vec[id_nzero, None]
        _c_24_vec= c_2_vec[id_nzero, None]/c_4_vec[id_nzero, None]
        _lower = lower[id_nzero, None]
        _upper = upper[id_nzero, None]


        res[id_nzero, None] = (_c_1_vec - _c_24_vec) * (_upper - _lower) + \
        integral_rational(_c_2_vec + _c_3_vec, (_c_2_vec * _c_5_vec)/_c_4_vec, -_c_4_vec, _c_4_vec, _c_5_vec, _lower, _upper)

    return F.relu(res) # slightly negative values can occur due to numerical errors: -1e-6

def integral_xi_rk(lower, upper, bin_idx, c_1, c_2, c_3, c_4, c_5):
    c_1_vec = c_1.gather(-1, bin_idx);
    c_2_vec = c_2.gather(-1, bin_idx); c_3_vec = c_3.gather(-1, bin_idx); 
    c_4_vec = c_4.gather(-1, bin_idx); c_5_vec = c_5.gather(-1, bin_idx);

    res = torch.zeros_like(c_1_vec)
    id_zero = (c_4_vec == 0)
    id_nzero = ~id_zero 
    
    if any(id_zero):
        #print("C_4 IS ZERO")
        _c_25_vec = c_2_vec[id_zero, None]/c_5_vec[id_zero, None]
        _c_35_vec = c_3_vec[id_zero, None]/c_5_vec[id_zero, None]
        _c_1_vec = c_1_vec[id_zero, None]
        _lower = lower[id_zero, None]
        _upper = upper[id_zero, None]
        res[id_zero, None] = (_c_25_vec/4.0) * (_upper.pow(4) - _lower.pow(4)) + (_c_35_vec/3.0) * (_upper.pow(3) - _lower.pow(3)) + (_c_1_vec/2.0) * (_upper.pow(2) - _lower.pow(2)) 
        
    if any(id_nzero):
        _c_1_vec = c_1_vec[id_nzero, None]
        _c_2_vec = c_2_vec[id_nzero, None]
        _c_3_vec = c_3_vec[id_nzero, None]
        _c_4_vec = c_4_vec[id_nzero, None]
        _c_5_vec = c_5_vec[id_nzero, None]
        _c_24_vec= c_2_vec[id_nzero, None]/c_4_vec[id_nzero, None]
        _lower = lower[id_nzero, None]
        _upper = upper[id_nzero, None]

        A = ((_c_2_vec * _c_5_vec)/_c_4_vec) + _c_3_vec + _c_2_vec
        B = ((_c_3_vec + _c_2_vec) * _c_5_vec) /_c_4_vec
        a = -_c_4_vec
        b = _c_4_vec
        c = _c_5_vec
        _c_24_vec = _c_2_vec/_c_4_vec


        res[id_nzero, None] = (_c_1_vec - _c_24_vec) * ((_upper**2 - _lower **2)/2.0) - ((_c_3_vec + _c_2_vec)/_c_4_vec) * (_upper - _lower) + \
            integral_rational(A, B, a, b, c, _lower, _upper)

    return F.relu(res) # slightly negative values can occur due to numerical errors: -1e-6
    #return res


