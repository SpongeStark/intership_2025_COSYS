import numpy as np

# CSF de Barten
# Il y a des valeurs par défaut sur la luminance moyenne et la taille du stimulus
# luminance_moyenne = 100 cd/m2
# stimulus = 10 degrés
def get_barten_csf(freq, mean_luminance=100, stimulus=10):
    a = (540 * (1 + 0.7 / mean_luminance)**-0.2) / (1 + (12/stimulus)*(1+freq/3)**-2)
    b = 0.3 * (1 + mean_luminance / 100)**0.15
    c = 0.06
    csf = a * freq * np.exp(-b * freq) * np.sqrt(1 + c*np.exp(b*freq))
    # la fonction renvoie la valeur de la CSF pour la fréquence "freq"
    return csf

# CSF de Barten
# Cette fonction renvoie un tableau avec dedans les valeurs de la CSF
# Il y a des valeurs par défaut sur la luminance moyenne et la taille du stimulus
# il faut donner le range de fréquences en cpd et un seuil
# je n'ai pas compris le rôle du seuil
# il y a luminance_moyenne = 100 cd/m2
# stimulus = 10 degrés
def get_barten_csf_squared(freq_range, mean_luminance=100, stimulus=10, threshold = 0.1):
    csf_squared = np.zeros_like(freq_range)
    for i in range(len(freq_range)):
        k = 1
        harmonic = 1
        while harmonic >= threshold:
            harmonic = (4 / (np.pi * k)) * get_barten_csf(k * freq_range[i], mean_luminance, stimulus)
            csf_squared[i] += harmonic
            k += 2
    return csf_squared