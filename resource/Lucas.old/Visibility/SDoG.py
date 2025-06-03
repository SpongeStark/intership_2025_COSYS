import numpy as np
from scipy.optimize import minimize

LAMBDA = 3

def get_fitting_sigma(frequencies):
    return (1 / (np.pi * frequencies)) * np.sqrt((np.log(LAMBDA) / (LAMBDA ** 2 - 1)))

def get_diff_fourier_gaussian(sigma, frequencies):
    exp_term_1 = np.exp(-2 * (np.pi ** 2) * (frequencies ** 2) * (sigma ** 2))
    exp_term_2 = np.exp(-2 * (np.pi ** 2) * (LAMBDA ** 2) * (frequencies ** 2) * (sigma ** 2))
    return exp_term_1 - exp_term_2

def get_slope_at_zero_crossing(sigma, frequencies, amplitude=1):
    K = get_diff_fourier_gaussian(sigma, frequencies)
    return 2 * np.pi * amplitude * K * frequencies

def get_fitting_weight(frequencies, function_to_mimic):
    sigma = function_to_mimic(frequencies)
    return function_to_mimic(frequencies) / (get_slope_at_zero_crossing(sigma, frequencies) + 0.0001)

def calculate_sum_dog(sigma_list, weight_list, frequencies):
    summed_dog = np.zeros_like(frequencies)
    for sigma, weight in zip(sigma_list, weight_list):
        slope_values = np.array([get_slope_at_zero_crossing(sigma, frequency) for frequency in frequencies])
        summed_dog += slope_values * weight
    return summed_dog

def error_function(params, f_range, curve_csf, dog_count):
    weight_list = params[:dog_count]
    sigma_list = params[dog_count:]
    sum_dog = calculate_sum_dog(sigma_list, weight_list, f_range)
    return np.sum((curve_csf - sum_dog)**2) / len(f_range)

def get_weights_regression(frequency_range, function_to_mimic, dog_count=6):
    initial_weights = np.ones(dog_count)
    initial_freqs = np.logspace(-1, 2, dog_count)
    initial_sigmas = get_fitting_sigma(initial_freqs)
    initial_params = np.concatenate([initial_sigmas, initial_weights])
    bounds = [(0.01, None)] * dog_count + [(0.01, 100)] * dog_count

    csf_curve = function_to_mimic(frequency_range)

    result = minimize(error_function, initial_params, args=(frequency_range, csf_curve, dog_count),
                      method='trust-constr', bounds=bounds)
    sigma_list = result.x[dog_count:]
    weight_list = result.x[:dog_count]

    return sigma_list, weight_list