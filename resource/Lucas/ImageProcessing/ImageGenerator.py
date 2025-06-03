import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


from ConversionFunctions import convert_angular_size_in_pixels, convert_pixels_in_angular_size, \
    convert_frequency_cpd_in_pixel_period, convert_pixel_period_in_frequency_cpd, convert_minutes_to_pixels


class Image:
    def __init__(self, file_path = None):
        self.image_array = None
        self.mean_intensity = 100

        if file_path:
            self.load_image(file_path)

    def load_image(self, file_path):
        self.image_array = cv.imread(file_path, cv.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        self.mean_intensity = np.mean(self.image_array)

    def load_txt(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        if len(lines) < 3:
            return np.array([])

        values = np.array(list(map(float, ' '.join(lines[2:]).replace(',', '.').split())))
        self.image_array = values.reshape((2036, 2464))
        self.mean_intensity = np.mean(self.image_array)

    def convert_into_linear_space(self):
        if self.image_array is not None:
            mask = self.image_array <= 0.04045
            self.image_array[mask] /= 12.92
            self.image_array[~mask] = ((self.image_array[~mask] + 0.055) / 1.055) ** 2.4

    def convert_into_gamma_space(self):
        if self.image_array is not None:
            mask = self.image_array <= 0.0031308
            self.image_array[mask] *= 12.92
            self.image_array[~mask] = 1.055 * (self.image_array[~mask] ** (1 / 2.4)) - 0.055

    def generate_sine_wave_img(self, img_width_pxl=None, img_width_degree=None, frequency=None, period_pxl=None,
                               img_ratio=1, distance_from_screen=60,lmoy=100.0,michelson=0.1):

        if img_width_pxl is None and img_width_degree is None:
            raise ValueError("Specify either img_width_pxl or img_width_degree.")

        if img_width_pxl is None:
            img_width_pxl = convert_angular_size_in_pixels(img_width_degree, distance_from_screen)
        if img_width_degree is None:
            img_width_degree = convert_pixels_in_angular_size(img_width_pxl, distance_from_screen)
        img_height_pxl = int(img_width_pxl / img_ratio)
        if frequency is None and period_pxl is None:
            raise ValueError("Specify either frequency or period_pxl.")
        if period_pxl is None:
            period_pxl = convert_frequency_cpd_in_pixel_period(frequency, distance_from_screen)
        if frequency is None:
            frequency = convert_pixel_period_in_frequency_cpd(period_pxl, distance_from_screen)

        x = np.arange(0, img_width_pxl)
        sine_wave = lmoy + michelson * lmoy * np.sin(2 * np.pi * x / period_pxl)
        sine_img = np.tile(sine_wave, (img_height_pxl, 1))
        print(f"[square rings] Distance: {distance_from_screen} cm, "
              f"Width: {img_width_degree:.2f} (Deg) & "
              f"{img_width_pxl:.2f} (Pix), "
              f"Lum moy: {lmoy:.2f} cd/m2, contrast: {michelson:.2f} " 
              f"Frequency: {frequency:.2f} cpd, "
              f"Period: {period_pxl:.2f} (Pix)")
        print(f"Lmin={lmoy - michelson * lmoy:.2f} et Lmax = {lmoy + michelson * lmoy:.2f}")

        self.image_array = sine_img
        nom = "images/output/sine_wave_image" + str(int(period_pxl)) + ".png"
        self.save_image(nom)
        self.mean_intensity = lmoy

    def generate_sine_circle_img(self, img_width_pxl=None, img_width_degree=None, frequency=None, period_pxl=None,
                                 img_ratio=1, distance_from_screen=60,lmoy=100.0,michelson=0.1):
        if img_width_pxl is None and img_width_degree is None:
            raise ValueError("Specify either img_width_pxl or img_width_degree.")

        if img_width_pxl is None:
            img_width_pxl = convert_angular_size_in_pixels(img_width_degree, distance_from_screen)
        if img_width_degree is None:
            img_width_degree = convert_pixels_in_angular_size(img_width_pxl, distance_from_screen)
        img_height_pxl = int(img_width_pxl / img_ratio)
        if frequency is None and period_pxl is None:
            raise ValueError("Specify either frequency or period_pxl.")

        if period_pxl is None:
            period_pxl = convert_frequency_cpd_in_pixel_period(frequency, distance_from_screen)
        if frequency is None:
            frequency = convert_pixel_period_in_frequency_cpd(period_pxl, distance_from_screen)

        center_x, center_y = img_width_pxl // 2, img_height_pxl // 2
        y, x = np.meshgrid(np.arange(img_height_pxl), np.arange(img_width_pxl), indexing="ij")
        distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        sine_img = lmoy + michelson * lmoy * np.cos(2 * np.pi * distances / period_pxl)

        print(f"[square rings] Distance: {distance_from_screen} cm, "
              f"Width: {img_width_degree:.2f} (Deg) & "
              f"{img_width_pxl:.2f} (Pix), "
              f"Lum moy: {lmoy:.2f} cd/m2, contrast: {michelson:.2f} " 
              f"Frequency: {frequency:.2f} cpd, "
              f"Period: {period_pxl:.2f} (Pix)")
        print(f"Lmin={lmoy - michelson * lmoy:.2f} et Lmax = {lmoy + michelson * lmoy:.2f}")

        self.image_array = sine_img
        self.mean_intensity = lmoy
        nom = "images/output/sine_circle_image" + str(int(period_pxl)) + ".png"
        self.save_image(nom)

    def generate_square_circle_img(self, img_width_pxl=None, img_width_degree=None, frequency=None, period_pxl=None,
                                   img_ratio=1, distance_from_screen=60,lmoy=100.0,michelson=0.1):

        if img_width_pxl is None and img_width_degree is None:
            raise ValueError("Specify either img_width_pxl or img_width_degree.")
        if img_width_pxl is None:
            img_width_pxl = convert_angular_size_in_pixels(img_width_degree, distance_from_screen)
        if img_width_degree is None:
            img_width_degree = convert_pixels_in_angular_size(img_width_pxl, distance_from_screen)

        img_height_pxl = int(img_width_pxl / img_ratio)

        if frequency is None and period_pxl is None:
            raise ValueError("Specify either frequency or period_pxl.")
        if period_pxl is None:
            period_pxl = convert_frequency_cpd_in_pixel_period(frequency, distance_from_screen)
        if frequency is None:
            frequency = convert_pixel_period_in_frequency_cpd(period_pxl, distance_from_screen)

        # on calcule la distance au centre de l'image
        center_x, center_y = img_width_pxl // 2, img_height_pxl // 2
        y, x = np.meshgrid(np.arange(img_height_pxl), np.arange(img_width_pxl), indexing="ij")
        distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

        # la valeur des pixels de l'image
        # par défaut les images ont une moyenne 1 et vont de 0 à 2
        # square_img = np.sign(np.cos(2 * np.pi * distances / period_pxl))
        square_img = lmoy + michelson * lmoy * np.sign(np.cos(2 * np.pi * distances / period_pxl))

        print(f"[square rings] Distance: {distance_from_screen} cm, "
              f"Width: {img_width_degree:.2f} (Deg) & "
              f"{img_width_pxl:.2f} (Pix), "
              f"Lum moy: {lmoy:.2f} cd/m2, contrast: {michelson:.2f} " 
              f"Frequency: {frequency:.2f} cpd, "
              f"Period: {period_pxl:.2f} (Pix)")
        print(f"Lmin={lmoy - michelson * lmoy:.2f} et Lmax = {lmoy + michelson * lmoy:.2f}")

        self.image_array = square_img
        self.mean_intensity = lmoy
        nom = "images/output/square_circle_image" + str(int(period_pxl)) + ".png"
        self.save_image(nom)

    def generate_square_wave_img(self, img_width_pxl=None, img_width_degree=None, frequency=None, period_pxl=None,
                                 img_ratio=1, distance_from_screen=60,lmoy=100.0,michelson=0.1):
        if img_width_pxl is None and img_width_degree is None:
            raise ValueError("Specify either img_width_pxl or img_width_degree.")

        if img_width_pxl is None:
            img_width_pxl = convert_angular_size_in_pixels(img_width_degree, distance_from_screen)
        if img_width_degree is None:
            img_width_degree = convert_pixels_in_angular_size(img_width_pxl, distance_from_screen)
        img_height_pxl = int(img_width_pxl / img_ratio)
        if frequency is None and period_pxl is None:
            raise ValueError("Specify either frequency or period_pxl.")

        if period_pxl is None:
            period_pxl = convert_frequency_cpd_in_pixel_period(frequency, distance_from_screen)
        if frequency is None:
            frequency = convert_pixel_period_in_frequency_cpd(period_pxl, distance_from_screen)

        x = np.arange(0, img_width_pxl)
        square_wave = lmoy + michelson * lmoy * np.sign(np.sin(2 * np.pi * x / period_pxl))
        square_img = np.tile(square_wave, (img_height_pxl, 1))
        print(f"[square rings] Distance: {distance_from_screen} cm, "
              f"Width: {img_width_degree:.2f} (Deg) & "
              f"{img_width_pxl:.2f} (Pix), "
              f"Lum moy: {lmoy:.2f} cd/m2, contrast: {michelson:.2f} " 
              f"Frequency: {frequency:.2f} cpd, "
              f"Period: {period_pxl:.2f} (Pix)")
        print(f"Lmin={lmoy - michelson * lmoy:.2f} et Lmax = {lmoy + michelson * lmoy:.2f}")

        self.mean_intensity = lmoy
        self.image_array = square_img
        nom = "images/output/square_wave_image" + str(int(period_pxl)) + ".png"
        self.save_image(nom)


    def tune_image_contrast_michelson(self, mean_luminance=100.0, contrast=0.1):
        self.image_array = self.image_array * contrast * mean_luminance + mean_luminance
        self.mean_intensity = mean_luminance

    def generate_target_img(self, img_width_pxl=1000, angular_size=10, luminance_background=100, luminance_target=120,
                            distance_from_screen=60):

        target_img = np.full((img_width_pxl, img_width_pxl), luminance_background, dtype=np.float64)
        center = (img_width_pxl // 2, img_width_pxl // 2)

        # pourquoi le +0.5 ?? sans doute pour éviter un diamètre nul
        # diameter = int(convert_minutes_to_pixels(angular_size, distance_from_screen) + 0.5)
        # diameter = int(convert_minutes_to_pixels(angular_size, distance_from_screen))
        # if diameter < 1: diameter = 1

        diameter_float = convert_minutes_to_pixels(angular_size, distance_from_screen)
        rayon_float = diameter_float/2.0
        for i in range (img_width_pxl):
            for j in range (img_width_pxl):
                r = np.sqrt((i-center[0])*(i-center[0]) + (j-center[1])*(j-center[1]))
                if(r<rayon_float):
                    target_img[i,j] = luminance_target

        # cv.circle(target_img, center, int(diameter / 2), [luminance_target], -1, lineType=cv.LINE_AA)

        self.mean_intensity = luminance_background
        self.image_array = target_img
        nom = "images/output/target_image" + str(int(angular_size)) + ".png"
        self.save_image(nom)

    def generate_aa_target_img(self, img_width_pxl=1000, angular_size=10, luminance_background=100, luminance_target=120,
                               distance_from_screen=60):
        target_img = np.full((img_width_pxl, img_width_pxl), luminance_background, dtype=np.float64)
        radius = convert_minutes_to_pixels(angular_size, distance_from_screen)
        center = (img_width_pxl // 2, img_width_pxl // 2)

        # on fabrique l'image à la main
        # mais c'est pas un disque, c'est un anneau
        for y in range(img_width_pxl):
            for x in range(img_width_pxl):
                dx = x - center[0]
                dy = y - center[1]
                distance_to_center = np.sqrt(dx ** 2 + dy ** 2)
                # distance (en pixels) du centre

                # radius est la taille du disque en pixels
                # si le pixel est pile à la distance "radius" du centre, on fait un truc soft
                if 0 < radius - distance_to_center < 1:
                    alpha = radius - distance_to_center
                    target_img[y, x] = luminance_background + alpha * (luminance_target - luminance_background)
                elif radius - distance_to_center > 0:
                    target_img[y, x] = luminance_target

        self.mean_intensity = luminance_background
        self.image_array = target_img
        nom = "images/output/aa_target_image" + str(int(angular_size)) + ".png"

        self.save_image(nom)

    def save_image(self, file_name: str):
        # normalized_array = cv.normalize(self.image_array, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
        # cv.imwrite(file_name, normalized_array)
        print("Save",file_name)
        cv.imwrite(file_name, self.image_array)