import sys
import os

sys.path.append(
    # 计算 `project_root/` 目录的路径
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
from ConversionFunctions import convert_angular_size_in_pixels
from Visibility.SDoG import get_fitting_sigma

LAMBDA = 3

class Filter:
    def __init__(self, distance_from_screen, sigma_list=None, weight_list = None, frequency=None, filter_size=None):
        self.distance_from_screen = distance_from_screen

        if sigma_list is not None and weight_list is not None:
            if len(sigma_list) != len(weight_list):
                raise ValueError("sigma_list and weight_list must have the same length")

        if sigma_list is None:
            sigma_list = [get_fitting_sigma(frequency)]
            if weight_list is None:
                weight_list = [1]

        sigma_list = convert_angular_size_in_pixels(np.array(sigma_list), distance_from_screen)

        if filter_size is None:
            filter_size = 2 * int(3 * LAMBDA * max(sigma_list)) + 1

        self.filter_size = filter_size

        self.filter_dx = np.zeros((self.filter_size, self.filter_size))
        self.filter_dy = np.zeros((self.filter_size, self.filter_size))

        if len(sigma_list) == 1:
            if weight_list is None:
                weight_list = [1]

        self.filter_array = np.zeros((self.filter_size, self.filter_size))
        for sigma, weight in zip(sigma_list, weight_list):
            self.filter_array += weight * self.generate_dog(sigma)

            dx, dy = self.generate_derived_dog(sigma)
            self.filter_dx += weight * dx
            self.filter_dy += weight * dy

    def generate_dog(self, sigma):
        sigma_minus = LAMBDA * sigma
        center = self.filter_size // 2

        y, x = np.meshgrid(np.arange(self.filter_size) - center, np.arange(self.filter_size) - center)

        g1 = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        g2 = (1 / (2 * np.pi * sigma_minus ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma_minus ** 2))

        dog_filter = g1 - g2
        dog_filter -= np.sum(dog_filter) / (self.filter_size ** 2)

        return dog_filter

    def generate_derived_dog(self, sigma):
        sigma_minus = LAMBDA * sigma
        center = self.filter_size // 2

        y, x = np.meshgrid(np.arange(self.filter_size) - center, np.arange(self.filter_size) - center)

        g1 = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        g2 = (1 / (2 * np.pi * sigma_minus ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma_minus ** 2))

        g_dx = -x / (sigma ** 2) * g1 + x / (sigma_minus ** 2) * g2
        g_dy = -y / (sigma ** 2) * g1 + y / (sigma_minus ** 2) * g2

        return g_dx, g_dy