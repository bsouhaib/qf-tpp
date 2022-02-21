from . import splines

class RationalQuadraticSpline:
    def __init__(self, left, right, bottom, min_qtail, max_qtail, num_bins, min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT, min_derivative=splines.DEFAULT_MIN_DERIVATIVE):
        self.left = left
        self.right = right
        self.bottom = bottom

        self.min_qtail = min_qtail
        self.max_qtail = max_qtail

        self.num_bins = num_bins

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

    def transform(self, inputs, cumwidths, widths, cumheights, heights, derivatives, inverse, compute_deriv, compute_sderiv):
        return splines.rational_quadratic_spline(inputs = inputs,
                                                num_bins = self.num_bins, 
                                                cumwidths = cumwidths, 
                                                widths = widths, 
                                                cumheights = cumheights,
                                                heights = heights, 
                                                derivatives = derivatives,
                                                inverse=inverse,
                                                compute_deriv = compute_deriv,
                                                compute_sderiv = compute_sderiv,
                                                min_bin_width = self.min_bin_width,
                                                min_bin_height = self.min_bin_height,
                                                min_derivative = self.min_derivative)

    def crps(self, y, all_tau_bar, inside_interval_mask, bin_idx, cumwidths, 
            widths, cumheights, heights, derivatives, input_cumwidths, input_bin_widths):
        return splines.crps(y, all_tau_bar, inside_interval_mask, bin_idx, cumwidths, 
                widths, cumheights, heights, derivatives, input_cumwidths, input_bin_widths)
    
    def integral_qf(self, cumwidths, widths, cumheights, heights, derivatives):
        return splines.integral_qf(cumwidths, widths, cumheights, heights, derivatives)