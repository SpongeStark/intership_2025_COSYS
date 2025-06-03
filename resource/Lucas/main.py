#############################################
# programme Lucas GOTTAR modifié
# (Roland Brémond 2025)
# calcul de visibilité avec une somme de DoG
#############################################

# Exemple d'utilisation d'os.path.abspath
import os  # Ajout de l'importation du module os
import sys  # Import nécessaire pour utiliser sys.path

new_path = "Visibility"
if new_path not in sys.path:
    sys.path.append(new_path)  # Ajout du chemin au PATH de Python
    print(f"Répertoire ajouté au PATH de Python : {new_path}")
# Afficher tout le PATH de Python
# print("Chemins actuels dans sys.path :")
# for chemin in sys.path:
#    print(chemin)



#imports
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

# fonction d'analyse d'une image de la base de données BIPED
# file : fichier d'entrée dans la base de données
# th1 : seuil de VL (exemple 1)
# th2 : seuil de VL (exemple 2)
# distance : distance de l'écran, en cm
def biped_dataset(file,th1,th2,distance):
    print("BIPED database, image",file,"thresholds:",th1,"and",th2,", d=",distance)
    grid = ImageGrid(2, 2)
    sdog_filter = Filter(distance_from_screen=distance,sigma_list=SIGMA_LIST,weight_list=WEIGHT_LIST)
    analyzer = ImageAnalyzer(sdog_filter)
    img = Image()
    img.load_image(f"BIPED/edges/imgs/train/rgbr/real/{file}.jpg")
    grid.add_image(img.image_array, "Original image", cmap="gray", row=0, column=0)
    img.convert_into_linear_space()
    analyzer.generate_visibility_map(img, method=GRADIENT)
    truth_img = Image()
    truth_img.load_image(f"BIPED/edges/edge_maps/train/rgbr/real/{file}.png")
    grid.add_image(truth_img.image_array, "Ground truth", cmap="gray", row=0, column=1)
    grid.add_image(analyzer.visibility_map > th1, f"Threshold = {th1}", cmap="gray", row=1, column=0)
    grid.add_image(analyzer.visibility_map > th2, f"Threshold = {th2}", cmap="gray", row=1, column=1)
    grid.show()

# Calcul de la carte de visibilité d'une image en luminance (ILMD)
# utile pour les données LUNNE mesurées par un ILMD
# d : distance de l'écran, en cm
def luminance_images(d):
    print("Distance from screen: d=",d,"cm")
    image_grid = ImageGrid(2, 4)
    sdog_filter = Filter(distance_from_screen=d,sigma_list=SIGMA_LIST,weight_list=WEIGHT_LIST)
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

# fonction de visualisation du modèle de CSF dans un graphe
# avec des points qui correspondent au calcul sur des images
# et le calcul d'un "similarity score"
# Lb : Luminance de fond en cd/m2
# Lt : luminance de cible en cd/m2 (pour les images de synthèse)
# d : distance d'observation des images sur l'écran, en cm
# image_width : taille des images de synthèse
# nb_DoG : nombre de DoG dans le modèle
def graph_csf(Lb,Lt,d,image_width,nb_ima):
    print("CSF computation with Lb=",Lb,", Lt=",Lt,"cd/m2 and d=",d," cm), with images size",image_width,'x',image_width)
    print("Test avec",nb_ima,"images")
    f_range = np.logspace(-1, 1.8, 1000)
    csf = get_barten_csf_squared(f_range, Lb)
    size_list = convert_frequency_cpd_to_minutes(f_range)
    adrian_threshold = get_luminance_target_threshold(size_list, Lb)
    graph = Graph("", "Spatial frequency", "Contrast sensitivity")
    graph.set_log_scale("xy")
    graph.add_curve(f_range, csf, "Barten's CSF")
    graph.add_curve(f_range, convert_luminance_threshold_to_contrast_sensitivity(adrian_threshold, Lb), "Adrian visibility")
    frequencies_samples = np.logspace(np.log10(0.5), np.log10(30.0), nb_ima)
    # Création d'un filtre SDoG pour des images vues à une distance de 500 cm
    sdog_filter = Filter(distance_from_screen=d,sigma_list=SIGMA_LIST,weight_list=WEIGHT_LIST)
    analyzer = ImageAnalyzer(sdog_filter)
    sensitivity_circles_list = []
    sensitivity_target_list = []
    for f in frequencies_samples:
        img = Image()
        img.generate_square_circle_img(image_width, img_ratio=1.0, frequency=f, distance_from_screen=d)
        analyzer.generate_visibility_map(img, method=GRADIENT)
        sensitivity_circles_list.append(analyzer.mean_visibility)
        img.generate_target_img(image_width, angular_size=convert_frequency_cpd_to_minutes(f),
                                luminance_background=Lb, luminance_target=Lt, distance_from_screen=d)
        analyzer.generate_visibility_map(img, method=GRADIENT)
        target_contrast = get_michelson_contrast(luminance_target=Lt, luminance_background=Lb)
        visibility = analyzer.mean_visibility
        sensitivity_target = visibility / target_contrast
        sensitivity_target_list.append(sensitivity_target)
    graph.add_curve(frequencies_samples, sensitivity_circles_list, curve_type=POINTS, color="blue")
    graph.add_curve(frequencies_samples, sensitivity_target_list, curve_type=POINTS, color="orange")
    graph.show()


def plot_SDoG(frequencies_samples,weights,Lb):
    print("Plot des DOG et comparaisons avec la CSF")
    f_range = np.logspace(-1, 1.8, 1000)
    csf = get_barten_csf_squared(f_range, Lb)
    size_list = convert_frequency_cpd_to_minutes(f_range)
    graph = Graph("", "Spatial frequency", "Contrast sensitivity")
    graph.set_log_scale("xy")
    graph.add_curve(f_range, csf, "Barten's CSF (référence)")
    graph.show()

# calcul de la carte de visibilité sur une image
# nom : nom de l'image
# d: distance de l'écran
def image_example(nom,d):
    nomFich = nom + ".png"
    print("Visibility map of",nomFich,"for d=",d,"cm")
    image_grid = ImageGrid(rows=2, columns=2) # Création de la grille d'images
    img = Image(nomFich) # Chargement de l'image
    img.convert_into_linear_space() # Conversion de l'image dans l'espace linéaire pour avoir une échelle de luminance

    # Création d'un filtre SDoG pour des images vues à une distance de d cm
    sdog_filter = Filter(distance_from_screen=d,sigma_list=SIGMA_LIST, weight_list=WEIGHT_LIST)

    analyzer = ImageAnalyzer(sdog_filter)
    analyzer.generate_visibility_map(img, method=GRADIENT)
    nom2 = nom + "_sdog.png"
    analyzer.save_image(nom2) # Sauvegarde des différentes images avec le nom "lena_sdog_[...].png
    img.convert_into_gamma_space() # Reconversion dans l'espace gamma pour l'affichage

    # Affichage des 4 images sur la grille
    image_grid.add_image(img.image_array, "Image originale", "gray", row=0, column=0)
    image_grid.add_image(analyzer.edge_localisation, "Localisation des contours", "gray", row=0, column=1)
    image_grid.add_image(analyzer.derived_filtered_img, "Gradient de l'image filtré par SDoG", "turbo", row=1, column=0)
    image_grid.add_image(analyzer.visibility_map, "Visibility map", "turbo", row=1, column=1, color_bar=True)
    image_grid.show()


# fonction de visualisation du modèle de CSF dans un graphe
# avec des points qui correspondent au calcul sur des images
# et le calcul d'un "similarity score"
# Création d'un filtre SDoG pour des images vues à une distance de 500 cm
# d : distance d'observation des images sur l'écran, en cm?
# Lt : luminance de la cible (en cd/m2)
# Lb : luminance de fond (en cd/m2)
# ima_width : taille des images (pixels)
# nb_samples : nombre d'images pour faire le calcul par SDoG
def graph_example(d=50.0,Lt=120,Lb=100.0,ima_width=2000,nb_samples=20,f_min=0.3,f_max=30):

    # contrast=get_michelson_contrast(Lt,Lb)
    contrast = (Lt - Lb)/(Lt + Lb)
    lmoy = (Lb + Lt)/2
    print(f"CSF with d={d:.2f} Lmoy={lmoy:.2f} Lt={Lt:.2f} et Lb={Lb:.2f} cd/m2, c={contrast:.2f}")
    print("calcul avec la SDOG par défaut stockée dans Parameters.py")

    # Chargement du banc de filtres
    sdog_filter = Filter(distance_from_screen=d,sigma_list=SIGMA_LIST,weight_list=WEIGHT_LIST)
    visibility_list_waves = []
    visibility_list_cibles = []
    # création du banc de filtres
    analyzer = ImageAnalyzer(sdog_filter)

    # choix des images qui vont servir à passer les SDoG
    # on commence par se mettre en log et échantillonner les fréquences
    # f_min = 0.3     # par défaut: 0.3 cpd
    # f_max = 30.0     # par défaut: 30 cpd
    # Echantillonnage des fréquences pour les images
    frequencies_samples = np.logspace(np.log10(f_min), np.log10(f_max), nb_samples)

    # pour chaque fréquence spatiale on choisit les tailles de cibles
    angular_sizes_samples = convert_frequency_cpd_to_minutes(frequencies_samples)
    print("tailles angulaires:",angular_sizes_samples)

    #on fabrique les images et on passe le banc de filtres dessus
    for f, s in zip(frequencies_samples, angular_sizes_samples):
        image = Image()
        print(f"image frequency: {f:.2f}%, angular size: {s:.2f}% minutes")

        # Création d'image de ima_width x ima_width
        # 1D sine waves
        # image.generate_sine_wave_img(ima_width, frequency=f, distance_from_screen=d)
        # 2D sine waves
        # image.generate_sine_circle_img(ima_width, frequency=f, distance_from_screen=d,lmoy=Lb,michelson=contrast)
        # 1D square waves
        # image.generate_square_wave_img(ima_width, frequency=f, distance_from_screen=d)
        # 2D square waves
        image.generate_square_circle_img(ima_width, frequency=f, distance_from_screen=d,lmoy=lmoy,michelson=contrast)
        # Cibles Adrian
        # image.generate_target_img(ima_width, angular_size=s, luminance_target=Lt, luminance_background=Lb,distance_from_screen=d)
        # Cible adrian ???
        # image.generate_aa_target_img(ima_width, angular_size=s, luminance_target=Lt, luminance_background=Lb,distance_from_screen=d)

        # calcul de la carte de visibilité
        analyzer.generate_visibility_map(image, method=GRADIENT)
        # on ajoute mean_visibility à la liste de visibilité (une par image)
        visibility_list_waves.append(analyzer.mean_visibility/contrast)

    # à tout hasard
    # for i in range(len(visibility_list_waves)):
    #    visibility_list_waves[i] /= 2

    # on fabrique les images de cibles simples et on passe le banc de filtres dessus
    for f, s in zip(frequencies_samples, angular_sizes_samples):
        image = Image()
        print(f"image frequency: {f:.2f}%, angular size: {s:.2f}% minutes")

        # Création d'image de ima_width x ima_width
        # Cibles Adrian
        print(f"Target {s:.2f} min.")
        # est-ce que la taille angulaire est s ou s/2 ??
        image.generate_target_img(ima_width, angular_size=s, luminance_target=Lt, luminance_background=Lb,distance_from_screen=d)
        # Cible adrian ???
        # image.generate_aa_target_img(ima_width, angular_size=s, luminance_target=Lt, luminance_background=Lb,distance_from_screen=d)

        # calcul de la carte de visibilité
        analyzer.generate_visibility_map(image, method=GRADIENT)
        # on ajoute mean_visibility à la liste de visibilité (une par image)
        visibility_list_cibles.append(analyzer.mean_visibility/contrast)

    # Calcul du score de précision entre la courbe et les points de la courbe bleue
    # On peut prendre comme référence soit Barten Squared (CSF carrée)...
    # ground_truth_array = np.log10(get_barten_csf_squared(frequencies_samples))
    ground_truth_array = np.log10(get_barten_csf_squared(frequencies_samples))

    # ...soit Barten normal, la CSF
    # ground_truth_array = np.log10(get_barten_csf(frequencies_samples))
    # mais pour comparer avec Adrian il faut prendre la version squared

    # calcul d'erreur entre a visibilité moyenne sur les images et la visiblité théorique d'après la CSF de Barten
    visibility_array = np.log10(np.array(visibility_list_waves))
    score_percentage = (1 - np.mean(np.abs((visibility_array - ground_truth_array) / ground_truth_array))) * 100
    print("Qualité du modèle SDoG par rapport à Barten:",score_percentage,"%")
    # print("Barten:",ground_truth_array)
    # print("Calcul:",visibility_array)

    # On trace le graphe avec les courbes et les points
    # Création du graphique avec titre et axes
    graph = Graph(title="SDoG model vs Adrian's model", x_label="f", y_label="sensitivity")
    # graph = Graph(title=f"Similarity score: {score_percentage:.2f}%", x_label="f", y_label="sensitivity")

    graph.set_log_scale("xy") # Mise en place d'une échelle logarithmique pour les axes
    # Ajout des points de visibilité moyenne de l'image
    graph.add_curve(frequencies_samples, visibility_list_waves, title="2D square waves", curve_type=POINTS, color="orange")
    graph.add_curve(frequencies_samples, visibility_list_cibles, title="2D targets", curve_type=POINTS,color="blue")

    # Ajout de la courbe CSF de Barten
    f_min = -1
    f_max = 1.5
    nb_pts = 1000
    frequencies_range = np.logspace(f_min,f_max, nb_pts) # Echantillon de fréquences pour la courbe
    Barten_csf = get_barten_csf_squared(frequencies_range)
    graph.add_curve(frequencies_range, Barten_csf, title="Barten CSF", curve_type=CURVE,color="orange")

    # calcul d'erreur entre a visibilité moyenne sur les images et la visiblité théorique d'après la CSF de Barten
    # graph.convert_to_adrian_space(Lb)
    angular_sizes_range = convert_frequency_cpd_to_minutes(frequencies_samples)
    ground_truth_array = get_luminance_target_threshold(angular_sizes_range, Lb)
    visibility_array = np.log10(np.array(visibility_list_cibles))
    # ground_truth_array = np.log10(get_visibility(frequencies_samples))
    print("VL calculé",visibility_array)
    print("VL GT",ground_truth_array)
    score_percentage = (1 - np.mean(np.abs((visibility_array - ground_truth_array) / ground_truth_array))) * 100
    print("Qualité du modèle SDoG par rapport à Adrian:",score_percentage,"%")

    # Pour visualiser ADRIAN :
    graph.convert_to_adrian_space(Lb)
    angular_sizes_range = convert_frequency_cpd_to_minutes(frequencies_range)
    target_visibility = get_luminance_target_threshold(angular_sizes_range, Lb)
    graph.add_curve(angular_sizes_range, target_visibility, title="Adrian model", curve_type=CURVE,color="blue")
    # graph.add_curve(angular_sizes_samples, convert_visibility_to_luminance_threshold(np.array(visibility_list), Lb, Lt), title="Adrian (images)", curve_type=POINTS, color="blue")
    graph.convert_to_csf_space(Lb)

    graph.show()

    # calcul du modèle SDoG
    # range_min : min des fréquences
    # range_max : max des fréquences
    # num_f: nombre de points sur la courbe
    # nb_DoG = nombre de DoG
    # Modèle: BARTEN (1990), avec la fonction get_barten_csf
def compute_model_SDoG(range_min,range_max,num_f,nb_DoG):
    frequency_range = np.logspace(range_min,range_max,num_f)
    function_to_mimic = get_barten_csf(frequency_range)
    sigmas,poids = get_weights_regression(frequency_range,get_barten_csf,nb_DoG)
    print("SIGMA_LIST =", sigmas)
    print("WEIGHT_LIST =",poids)

    # essayons de l'afficher
    graph = Graph(title="SDoG model", x_label="spatial frequency (cpd)", y_label="sensitivity")
    graph.set_log_scale("xy")
    graph.set_ylim(0.1,1000)
    # Barten
    graph.add_curve(frequency_range, function_to_mimic, title="Barten CSF", curve_type=CURVE,color="orange")
    # créer une DoG avec un poids
    sum_slope_at_zero = np.zeros_like(frequency_range)
    for i, (sigma, weight) in enumerate(zip(sigmas,poids)):
        slote_at_zero = get_slope_at_zero_crossing(sigma, frequency_range, weight)
        # Pour tracer chaque courbe individuellement :
        graph.add_curve(frequency_range, slote_at_zero, f"DoG {i + 1}", curve_type=CURVE)
        sum_slope_at_zero += slote_at_zero
    # Pour tracer la somme des DoG :
    graph.add_curve(frequency_range, sum_slope_at_zero, f"SDoG", curve_type=CURVE)
    graph.show()

# image_width = 2000  # taille des images de synthèse (Lucas)
# nb_DoG = 15         # nombre de DoG dans le modèle (Lucas)
# nb_samples = 20     # nombre de points sur la courbe (Lucas)
if __name__ == "__main__":
    # variables
    nb_DoG = 7              # Nombre de DoG dans le modèle SDoG
    range_min = -1.0          # fréquence min pour calculer le modèle
    range_max = 2.0         # fréquence max pour calculer le modèle
    num_f = 200             # nb de points pour calculer le modèle
    Lb = 100.0              # Luminance de fond en cd/m2
    Lt = 120.0              # luminance de cible en cd/m2 (pour les images de synthèse)
    distance = 500.0        # distance d'observation des images sur l'écran, en cm
    image_width = 1000      # taille des images de synthèse. Par défaut: 2000
    nb_samples = 10         # nombre de points sur la courbe. Par défaut: 20
    f_min = 0.3             # par défaut: 0.3 cpd
    f_max = 30              # par défaut: 30 cpd
    th1 = 60.0              # seuil de VL (exemple 1)
    th2 = 80.0              # seuil de VL (exemple 2)
    dir = "images/real/"    # répertoire des images
    nomFich = "lena"        # nom du fichier image (.png)
    file = "RGB_002"        # fichier d'entrée dans la base de données

    # Etape 0: Tentative pour créer une SDoG
    # compute_model_SDoG(range_min,range_max,num_f,nb_DoG)
    
    # Etape 1: CSF de Barten: fabrication du modèle SDoG et test avec des images
    graph_csf(Lb,Lt,distance,image_width,nb_samples)

    # etape 2: plot de la CSF et des valeurs calculés avec la SDoG enregistrée dans Parameters.py
    # graph_example(distance,Lt,Lb,image_width,nb_samples,f_min,f_max)

    # Etape 3: exemple de carte de visibilité
    # distance = 30.0           # distance d'observation de l'écran
    # image_example(dir + nomFich,distance)

    # Etape 4: exemple d'application sur des images en luminance (LUNNE)
    # distance = 30.0      # distance de l'écran, en cm
    # luminance_images(distance)

    # Etape 5: test sur des images de la base de données BIPED
    # distance = 50.0           # distance de l'écran, en cm
    # biped_dataset(file, th1, th2, distance)

    print("The END")