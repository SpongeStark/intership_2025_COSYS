import numpy as np

from Parameters import SCREEN_SIZE_PXL, SCREEN_SIZE_CM

MINUTES = 60

def convert_minutes_to_frequency_cpd(angular_size):
    return MINUTES / (2 * angular_size)

def convert_frequency_cpd_to_minutes(frequency):
    return MINUTES / (2 * frequency)

def convert_minutes_to_pixels(angular_size, distance_from_screen):
    return convert_angular_size_in_pixels(angular_size / 60, distance_from_screen)

def convert_angular_size_in_pixels(angular_size, distance_from_screen):
    return 2 * distance_from_screen * np.tan(0.5 * angular_size * np.pi / 180) * (SCREEN_SIZE_PXL / SCREEN_SIZE_CM)

def convert_pixels_in_angular_size(pixel_width, distance_from_screen):
    return 2 * np.arctan(pixel_width * (SCREEN_SIZE_CM / SCREEN_SIZE_PXL) / (2 * distance_from_screen)) * 180 / np.pi

def convert_frequency_cpd_in_pixel_period(frequency_cpd, distance_from_screen):
    period = convert_angular_size_in_pixels(1, distance_from_screen) / frequency_cpd
    return period

# On a la visibilité d'une cible, au sens d'Adrian, et sa taille.
# On veut le seuil de perception qui va avec, toujours d'après Adrian
# Qu'est-ce que ça veut dire exactement ?
def convert_visibility_to_luminance_threshold(visibility, luminance_background, luminance_target):
    actual_contrast = (luminance_target - luminance_background) / (luminance_target + luminance_background)
    th_contrast = actual_contrast / visibility
    return (luminance_background * ((th_contrast + 1) / (1 - th_contrast))) - luminance_background

def convert_luminance_threshold_to_contrast_sensitivity(luminance_threshold, luminance_background):
    sensitivity_to_contrast = (luminance_threshold + 2*luminance_background) / luminance_threshold

    return sensitivity_to_contrast

def convert_to_adrian_space(frequencies, sensitivity, luminance_background):
    frequencies = np.array(frequencies)
    sensitivity = np.array(sensitivity)

    size_list = convert_frequency_cpd_to_minutes(frequencies)
    luminance_threshold = (2 * luminance_background) / (sensitivity - 1)
    luminance_threshold[luminance_threshold < 0] = np.max(luminance_threshold)
    return size_list, luminance_threshold

def convert_to_csf_space(size_list, luminance_threshold, luminance_background):
    # Assurez-vous que les entrées sont bien des numpy arrays
    size_list = np.array(size_list)
    luminance_threshold = np.array(luminance_threshold)

    frequencies = convert_minutes_to_frequency_cpd(size_list)
    sensitivity = convert_luminance_threshold_to_contrast_sensitivity(luminance_threshold, luminance_background)

    return frequencies, sensitivity

def convert_pixel_period_in_frequency_cpd(pixel_period, distance_from_screen):
    frequency = 1 / (convert_pixels_in_angular_size(1, distance_from_screen) * pixel_period)
    return frequency

def get_michelson_contrast(luminance_target, luminance_background):
    return (luminance_target - luminance_background) / (luminance_target + luminance_background)

def get_target_luminance(luminance_background, contrast):
    return (contrast * luminance_background + luminance_background) / (1 - contrast)

