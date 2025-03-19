import numpy as np
import cv2 as cv

from ..ConversionFunctions import convert_pixels_in_angular_size
from .ConvolutionFilter import Filter
from .ImageGenerator import Image

ZERO_CROSSING = 0
GRADIENT = 1


class ImageAnalyzer:
    def __init__(self, convolution_filter:Filter):
        self.filter = convolution_filter

        self.image_array = None

        self.filtered_img = None
        self.derived_filtered_img = None
        self.edge_localisation = None
        self.visibility_map = None

        self.pixel_size_degree = convert_pixels_in_angular_size(1, self.filter.distance_from_screen)

        self.mean_intensity = 1

        self.mean_visibility = 0

    def get_edge_localisation(self, method, dx=None, dy=None):
        epsilon = 0
        if method == GRADIENT:
            self.edge_localisation = np.zeros_like(dy)
            height, width = self.derived_filtered_img.shape

            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    self.edge_localisation[y, x] = 0
                    if dy[y, x] != 0.0:
                        gradient_angle = np.arctan2(dy[y, x], dx[y, x]) * 180 / np.pi

                        y1, y2, x1, x2 = y, y, x, x

                        if (-22.5 <= gradient_angle < 22.5) or (157.5 <= gradient_angle <= 180) or (
                                -180 <= gradient_angle < -157.5):
                            (y1, y2) = (y - 1, y + 1) # Horizontal (0°)
                        elif (22.5 <= gradient_angle < 67.5) or (-157.5 <= gradient_angle < -112.5):
                            (y1, y2, x1, x2) = (y - 1, y + 1, x - 1, x + 1) # Diagonal 45° (upward)
                        elif (67.5 <= gradient_angle < 112.5) or (-112.5 <= gradient_angle < -67.5):
                            (x1, x2) = (x - 1, x + 1)  # Vertical (90°)
                        elif (112.5 <= gradient_angle < 157.5) or (-67.5 <= gradient_angle < -22.5):
                            (y1, y2, x1, x2) = (y - 1, y + 1, x + 1, x - 1)  # Diagonal -45° (downward)

                        if(self.derived_filtered_img[y1, x1] + epsilon < self.derived_filtered_img[y, x] and
                            self.derived_filtered_img[y2, x2] + epsilon < self.derived_filtered_img[y, x]):
                            self.edge_localisation[y, x] = 1

        else:
            zero_crossing_image = np.zeros_like(self.filtered_img)

            sign_img = np.sign(self.filtered_img)

            crossing_vertical = (np.roll(sign_img, 1, axis=0) * np.roll(sign_img, -1, axis=0) < 0)
            crossing_horizontal = (np.roll(sign_img, 1, axis=1) * np.roll(sign_img, -1, axis=1) < 0)

            zero_crossing_image[crossing_vertical] = 1
            zero_crossing_image[crossing_horizontal] = 1

            self.edge_localisation = zero_crossing_image

    def compute_mean_visibility(self):
        epsilon = 10

        border_fraction = 0.2
        height, width = self.visibility_map.shape

        top = int(height * border_fraction)
        bottom = height - top
        left = int(width * border_fraction)
        right = width - left

        cropped_visibility_map = self.visibility_map[top:bottom, left:right]
        self.mean_visibility = np.mean(cropped_visibility_map[cropped_visibility_map > epsilon])

    def generate_visibility_map(self, img:Image, method=GRADIENT, edge_map=None):
        preprocessed_image = img.image_array / np.mean(img.mean_intensity)
        self.filtered_img = cv.filter2D(preprocessed_image, -1, self.filter.filter_array, borderType=cv.BORDER_REPLICATE)

        image_dx = cv.filter2D(preprocessed_image, -1, self.filter.filter_dx, borderType=cv.BORDER_REPLICATE) / self.pixel_size_degree
        image_dy = cv.filter2D(preprocessed_image, -1, self.filter.filter_dy, borderType=cv.BORDER_REPLICATE) / self.pixel_size_degree

        self.derived_filtered_img = np.hypot(image_dx, image_dy)
        if edge_map is not None:
            self.edge_localisation = edge_map
        else:
            self.get_edge_localisation(method, image_dx, image_dy)
        self.visibility_map = np.where(self.edge_localisation == 1, self.derived_filtered_img, 0)

        self.compute_mean_visibility()

    def save_image(self, file_name: str):
        derived_filtered_normalized_array = cv.normalize(self.derived_filtered_img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        edges_normalized_array = cv.normalize(self.edge_localisation, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        visibility_normalized_array = cv.normalize(self.visibility_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        cv.imwrite(file_name + "_filtered_derived.png", derived_filtered_normalized_array)
        cv.imwrite(file_name + "_edges.png", edges_normalized_array)
        cv.imwrite(file_name + "_visibility_map.png", visibility_normalized_array)

        print("4 images transformed with SDoG operator saved !")

    def save_visibility_map(self, file_name: str):
        visibility_normalized_array = cv.normalize(self.visibility_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        cv.imwrite(file_name + ".png", visibility_normalized_array)

        print("Visibility map saved !")