from ConversionFunctions import convert_frequency_cpd_to_minutes, convert_luminance_threshold_to_contrast_sensitivity, \
    get_michelson_contrast, convert_visibility_to_luminance_threshold
from ImageProcessing.ImageAnalyzer import ImageAnalyzer, GRADIENT, ZERO_CROSSING
from Parameters import WEIGHT_LIST, SIGMA_LIST
from Plotting.PltGraph import Graph, CURVE, POINTS
from Plotting.PltImages import ImageGrid

from ImageProcessing.ImageGenerator import Image

from ImageProcessing.ConvolutionFilter import Filter

from Visibility.BartenCSF import *
from Visibility.Adrian import *
from Visibility.SDoG import *

def biped_dataset():
    grid = ImageGrid(2, 2)

    sdog_filter = Filter(distance_from_screen=50,
                         sigma_list=SIGMA_LIST,
                         weight_list=WEIGHT_LIST)
    analyzer = ImageAnalyzer(sdog_filter)

    file = "RGB_002"
    img = Image()
    img.load_image(f"BIPED/edges/imgs/train/rgbr/real/{file}.jpg")
    grid.add_image(img.image_array, "Original image", cmap="gray", row=0, column=0)

    img.convert_into_linear_space()
    analyzer.generate_visibility_map(img, method=GRADIENT)

    truth_img = Image()
    truth_img.load_image(f"BIPED/edges/edge_maps/train/rgbr/real/{file}.png")
    grid.add_image(truth_img.image_array, "Ground truth", cmap="gray", row=0, column=1)

    th1 = 60.0
    grid.add_image(analyzer.visibility_map > th1, f"Threshold = {th1}", cmap="gray", row=1, column=0)

    th2 = 80.0
    grid.add_image(analyzer.visibility_map > th2, f"Threshold = {th2}", cmap="gray", row=1, column=1)

    grid.show()

def luminance_images():
    image_grid = ImageGrid(2, 4)

    d = 30
    sdog_filter = Filter(distance_from_screen=d,
                         sigma_list=SIGMA_LIST,
                         weight_list=WEIGHT_LIST,
                         )

    analyzer = ImageAnalyzer(sdog_filter)

    img = Image()

    file_name_list = ["ZA_FER_lettres_50%", "ZB_FER_lettres_10%", "5_c", "3_c"]

    for i in range(len(file_name_list)):
        img.load_txt(f"images/real/{file_name_list[i]}.txt")
        analyzer.generate_visibility_map(img)
        img.convert_into_gamma_space() # Uniquement pour l'affichage
        image_grid.add_image(img.image_array, "Luminance image", "gray", 0, i, vmin = 0, vmax=1)
        image_grid.add_image(analyzer.visibility_map, "Edge detection", "turbo", 1, i, vmin=0, vmax=100, color_bar=True)

    image_grid.show()

def graph_csf():
    f_range = np.logspace(-1, 1.8, 1000)

    csf = get_barten_csf_squared(f_range, 100)

    size_list = convert_frequency_cpd_to_minutes(f_range)
    adrian_threshold = get_luminance_target_threshold(size_list, 100)

    graph = Graph("", "Frequency", "Sensitivity to contrast")
    graph.set_log_scale("xy")

    graph.add_curve(f_range, csf, "CSF Barten")
    graph.add_curve(f_range, convert_luminance_threshold_to_contrast_sensitivity(adrian_threshold, 100), "Adrian visibility")

    frequencies_samples = np.logspace(np.log10(0.5), np.log10(30.0), 15)

    # Création d'un filtre SDoG pour des images vues à une distance de 500 cm
    d = 500
    sdog_filter = Filter(distance_from_screen=d,
                         sigma_list=SIGMA_LIST,
                         weight_list=WEIGHT_LIST)
    analyzer = ImageAnalyzer(sdog_filter)
    sensitivity_circles_list = []
    sensitivity_target_list = []

    for f in frequencies_samples:
        img = Image()

        img.generate_square_circle_img(2000, img_ratio=1.0, frequency=f, distance_from_screen=d)
        analyzer.generate_visibility_map(img, method=GRADIENT)
        sensitivity_circles_list.append(analyzer.mean_visibility)

        img.generate_target_img(2000, angular_size=convert_frequency_cpd_to_minutes(f),
                                luminance_background=100, luminance_target=200, distance_from_screen=d)
        analyzer.generate_visibility_map(img, method=GRADIENT)

        target_contrast = get_michelson_contrast(luminance_target=200, luminance_background=100)
        visibility = analyzer.mean_visibility

        sensitivity_target = visibility / target_contrast

        sensitivity_target_list.append(sensitivity_target)

    graph.add_curve(frequencies_samples, sensitivity_circles_list, curve_type=POINTS, color="blue")
    graph.add_curve(frequencies_samples, sensitivity_target_list, curve_type=POINTS, color="orange")
    graph.show()

def graph_example():
    # Création d'un filtre SDoG pour des images vues à une distance de 500 cm
    d = 500
    sdog_filter = Filter(distance_from_screen=d,
                         sigma_list=SIGMA_LIST,
                         weight_list=WEIGHT_LIST)

    visibility_list = []
    analyzer = ImageAnalyzer(sdog_filter)

    Lt = 200
    Lb = 100
    frequencies_samples = np.logspace(np.log10(0.3), np.log10(30.0), 20) # Echantillon de fréquences pour les images
    angular_sizes_samples = convert_frequency_cpd_to_minutes(frequencies_samples)
    for f, s in zip(frequencies_samples, angular_sizes_samples):
        # Création d'image de 2000x2000 de sinusoïdes de différentes fréquences vues à une distance de 500 cm
        sine_img = Image()
        # sine_img.generate_square_circle_img(2000, frequency=f, distance_from_screen=d)
        sine_img.generate_target_img(2000, angular_size=s, luminance_target=Lt, luminance_background=Lb,
                                     distance_from_screen=d)

        analyzer.generate_visibility_map(sine_img, method=GRADIENT)

        visibility_list.append(analyzer.mean_visibility)

    # Calcul du score de précision entre la courbe et les points
    ground_truth_array = np.log10(get_barten_csf_squared(frequencies_samples))
    visibility_array = np.log10(np.array(visibility_list))
    score_percentage = (1 - np.mean(np.abs((visibility_array - ground_truth_array) / ground_truth_array))) * 100

    graph = Graph(title=f"Similarity score : {score_percentage:.2f}%", x_label="", y_label="") # Création du graphique avec titre et axes
    graph.set_log_scale("xy") # Mise en place d'une échelle logarithmique pour les axes

    # Ajout des points de visibilité moyenne de l'image
    graph.add_curve(frequencies_samples, visibility_list, title="", curve_type=POINTS, color="orange")

    # Ajout de la courbe CSF de Barten
    frequencies_range = np.logspace(-1, 1.7, 1000) # Echantillon de fréquences pour la courbe
    barten_csf = get_barten_csf_squared(frequencies_range)
    graph.add_curve(frequencies_range, barten_csf, title="", curve_type=CURVE)

    # Pour combiner avec ADRIAN :

    # graph.convert_to_adrian_space(100)
    #
    # angular_sizes_range = convert_frequency_cpd_to_minutes(frequencies_range)
    # target_visibility = get_luminance_target_threshold(angular_sizes_range, 100)
    #
    # graph.add_curve(angular_sizes_range, target_visibility, title="", curve_type=CURVE)

    # graph.add_curve(angular_sizes_samples, convert_visibility_to_luminance_threshold(np.array(visibility_list), Lb, Lt), title="", curve_type=POINTS, color="orange")

    # graph.convert_to_csf_space(100)

    graph.show()

def image_example():
    image_grid = ImageGrid(rows=2, columns=2) # Création de la grille d'images

    img = Image("images/real/lena.png") # Chargement de l'image
    img.convert_into_linear_space() # Conversion de l'image dans l'espace linéaire pour avoir une échelle de luminance

    # Création d'un filtre SDoG pour des images vues à une distance de 30 cm
    d = 30
    sdog_filter = Filter(distance_from_screen=d,
                         sigma_list=SIGMA_LIST,
                         weight_list=WEIGHT_LIST)

    analyzer = ImageAnalyzer(sdog_filter)
    analyzer.generate_visibility_map(img, method=GRADIENT)
    analyzer.save_image("lena_sdog") # Sauvegarde des différentes images avec le nom "lena_sdog_[...].png

    img.convert_into_gamma_space() # Reconversion dans l'espace gamma pour l'affichage

    # Affichage des 4 images sur la grille
    image_grid.add_image(img.image_array, "Image originale", "gray", row=0, column=0)
    image_grid.add_image(analyzer.edge_localisation, "Localisation des contours", "gray", row=0, column=1)
    image_grid.add_image(analyzer.derived_filtered_img, "Gradient de l'image filtré par SDoG", "turbo", row=1, column=0)
    image_grid.add_image(analyzer.visibility_map, "Carte de visibilité", "turbo", row=1, column=1, color_bar=True)

    image_grid.show()

if __name__ == "__main__":
    image_example()
    # graph_example()
    # luminance_images()
    # biped_dataset()