import numpy as np

def get_phi(L_b):
    if L_b >= 0.6:
        return np.log10(4.1925* (L_b**0.1556))+ 0.1684 * (L_b**0.5867)
    elif L_b <= 0.00418:
        return 10**(0.028 + 0.173 * np.log10(L_b))
    else:
        return 10**(-0.072 + 0.3372*np.log10(L_b) + 0.0866*((np.log10(L_b))**2))

def get_luminance_function(L_b):
    if L_b >= 0.6:
        return 0.05946 * (L_b**0.466)
    elif L_b <= 0.00418:
        return 10 ** (-0.891 + 0.5275 * np.log10(L_b) + 0.0227 * (np.log10(L_b))**2)
    else:
        return 10 ** (-1.256 + 0.319 * np.log10(L_b))

def get_contrast_polarity_factor(alpha, L_b):
    get_exposure_time_vectorized = np.vectorize(get_exposure_time_influence)
    L_pos2 = get_exposure_time_vectorized(alpha, L_b, 2)

    beta = 0.6 * L_b**(-0.1488)

    if L_b >= 0.1:
        m = 10**(-10**(-0.125 * (np.log10(L_b + 1) ** 2 + 0.0245)))
    else:
        m = 10**(-10**(-0.075 * (np.log10(L_b + 1) ** 2 + 0.0245)))

    return 1 - ((m * alpha**(-beta)) / (2.4 * L_pos2))

def get_exposure_time_influence(alpha, L_b, t):
    a_alpha = 0.36 - 0.0972 * (((np.log10(alpha) + 0.523)**2) /
                     (((np.log10(alpha) + 0.523)**2) - 2.513 * (np.log10(alpha) + 0.523) + 2.7895))

    a_Lb = 0.355 - 0.1217 * (((np.log10(L_b) + 6)**2) /
                             (((np.log10(L_b) + 6) ** 2) - 10.4 * (np.log10(L_b) + 6) + 52.28))

    a = np.sqrt(a_alpha**2 + a_Lb**2) / 2.1

    return (a + t) / t

def get_age_factor(age):
    if age <= 23:
        return 1.0
    elif age <= 64:
        return ((age - 19)**2 / 2160) + 0.99
    else:
        return ((age - 56.6)**2 / 116.3) + 1.43

def get_luminance_target_threshold(angular_size, luminance_bg = 1000,
                                   age=23, exposure_time=2.0):
    get_phi_vectorized = np.vectorize(get_phi)
    get_luminance_vectorized = np.vectorize(get_luminance_function)
    get_exposure_time_vectorized = np.vectorize(get_exposure_time_influence)
    get_contrast_polarity_vectorized = np.vectorize(get_contrast_polarity_factor)
    get_age_factor_vectorized = np.vectorize(get_age_factor)

    phi = get_phi_vectorized(luminance_bg)
    L = get_luminance_vectorized(luminance_bg)
    Fcp = get_contrast_polarity_vectorized(angular_size, luminance_bg)
    Fet = get_exposure_time_vectorized(angular_size, luminance_bg, exposure_time)
    AF = get_age_factor_vectorized(age)

    delta_L = ((phi / angular_size) + L)**2 * Fet * AF # * Fcp

    return delta_L

def get_visibility(angular_size, luminance_target = 150, luminance_bg=100, delta_luminance=None):
    if delta_luminance is None:
        delta_luminance = get_luminance_target_threshold(luminance_bg=luminance_bg, angular_size=angular_size)
    return (luminance_target - luminance_bg) / delta_luminance