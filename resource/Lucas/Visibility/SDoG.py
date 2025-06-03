import numpy as np
from scipy.optimize import minimize

# Fonction CSF de Barten
from .BartenCSF import *

# ratio entre la grande et la petite gaussienne dans les DoG
LAMBDA = 3.0

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
    initial_freqs = np.logspace(-1, 2, dog_count)
    initial_weights = np.ones(dog_count)
    initial_sigmas = get_fitting_sigma(initial_freqs)
    # initial_sigmas = [0.29070475, 0.11312207, 0.89904843, 0.02748888, 0.05124298, 0.01315371]
    # initial_weights = [37.85748477, 32.23029905, 45.770649,   12.06325247, 24.9225947,   2.85437407]
    initial_params = np.concatenate([initial_sigmas, initial_weights])
    bounds = [(0.01, None)] * dog_count + [(0.01, 100)] * dog_count

    csf_curve = function_to_mimic(frequency_range)
    # csf_curve = get_barten_csf_squared(frequency_range)
    result = minimize(error_function, initial_params, args=(frequency_range, csf_curve, dog_count),
                      method='trust-constr', bounds=bounds)
    sigma_list = result.x[dog_count:]
    weight_list = result.x[:dog_count]
    print("erreur=",error_function(result.x, frequency_range, csf_curve, dog_count))

    # initial_sigmas = [0.25076792, 0.0981563, 0.78554953, 0.04379596, 0.01140447, 0.02189041]
    # initial_weights = [37.83331329, 32.03847274, 46.84298563, 25.18805022, 1.51597104, 7.92882045]
    # initial_params = np.concatenate([initial_sigmas, initial_weights])
    # print("erreur2=", error_function(initial_params, frequency_range, csf_curve, len(initial_sigmas)))

    return sigma_list, weight_list