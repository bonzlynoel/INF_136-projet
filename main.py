# Importation des bibliothèques nécessaires
from visualisation import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

if __name__ == '__main__':
    # Ceci est une procédure de test pour exécuter l'ensemble des sous-programmes.

    # Calcul des coordonnées pour un objet "clou" et visualisation de ces points
    coords_clou = calculer_coordonnees_clou(3, 10, 1, 0.75, 2)
    visualiser_points_clou(coords_clou)

    # Application de transformations (réflexion, rotation, inclinaison) sur les points du clou
    # et visualisation des résultats de ces transformations
    reflected_points_list, rotated_points_list, inclin_points_list = appliquer_transormation_clou(coords_clou, (0,0), 30, 'x', 20, 'x')
    visualiser_transformations_clou(reflected_points_list, rotated_points_list, inclin_points_list)

    # Chemin des images d'origine et en niveaux de gris
    path_image_orig = 'image_couleur.jpg'
    path_image_ng = 'image_niveaux_de_gris.jpg'

    # Conversion d'une image couleur en niveaux de gris et visualisation des deux images
    rgb_to_gry(path_image_orig, path_image_ng)
    visualiser_image_couleur_ng(path_image_orig, path_image_ng)

    # Ouverture de l'image en niveaux de gris et conversion en tableau NumPy
    img = Image.open(path_image_ng).convert('L')
    img_array = np.array(img)

    # Application de transformations sur l'image et stockage des résultats
    image_trasf_1 = appliquer_transformation_1(img_array)
    image_trasf_2 = appliquer_transformation_2(img_array, radius=2)

    # Création d'une liste pour stocker les images et les titres correspondants
    images = [img_array, image_trasf_1, image_trasf_2]
    titles = ['Image en NG', 'Image après transformation 1', 'Image après transformation 2']

    # Création d'une figure avec plusieurs sous-graphiques pour afficher les images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 ligne, 3 colonnes

    # Boucle pour afficher chaque image dans les sous-graphiques
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')  # Désactivation des axes pour une meilleure visibilité

    plt.show()  # Affichage de la figure contenant les images

    # Calcul d'un histogramme pour l'image transformée et regroupement des points
    w = 3
    tab_histo = calculer_histogramme(image_trasf_2, w)
    labels2 = regrouper_points(tab_histo)

    # Redimensionnement et affichage de l'image segmentée
    segmented_image = labels2.reshape(img_array.shape[0] - w + 1, img_array.shape[1] - w + 1)
    plt.imshow(segmented_image, cmap='gray')
    plt.title("Image Segmentée")
    plt.show()
