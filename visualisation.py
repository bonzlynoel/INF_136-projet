# Importation des modules nécessaires pour la manipulation des coordonnées,
# le traitement des images, la segmentation et la manipulation des histogrammes.
# matplotlib.pyplot est utilisé pour la visualisation graphique.
from coordonnees_clou import *
from traitement_image import *
from segmentation import *
from manupulation_histogramme import *
import matplotlib.pyplot as plt

# Définition des constantes qui représentent les dimensions spécifiques du clou.
__A = 3
__B = 10
__C = 1
__D = 0.75
__E = 2

# Calcul des coordonnées du clou basé sur les dimensions définies ci-dessus.
__COORDS_CLOU = calculer_coordonnees_clou(__A, __B, __C, __D, __E)

# Paramètres pour les transformations appliquées au clou.
__CENTER_ROT = (0,0)  # Centre de rotation
__ANGLE_ROT = 30      # Angle de rotation
__DIR_INCL = 'x'      # Axe d'inclinaison
__ANGLE_INCL = 20     # Angle d'inclinaison
__AXE_REFLEX = 'x'    # Axe de réflexion

# Application des transformations (réflexion, rotation, inclinaison) sur le clou
# et stockage des coordonnées transformées.
__REFLECTED_COORD, __ROTATED_COORD, __INCLIN_COORD = appliquer_transormation_clou(__COORDS_CLOU, __CENTER_ROT, __ANGLE_ROT, __DIR_INCL, __ANGLE_INCL, __AXE_REFLEX)

# Chemins vers les images originale et en niveaux de gris.
__PATH_IMAGE_ORIG = 'image_couleur.jpg'
__PATH_IMAGE_NG = 'image_niveaux_de_gris.jpg'

# Fonction pour visualiser les points du clou.
def visualiser_points_clou(coordonnees_et_noms):

    # Création d'une figure pour la visualisation graphique des points du clou.
    plt.figure(figsize=(10, 4))
    for nom, (x, y) in coordonnees_et_noms:
        plt.plot(x, y, marker='o')  # Marquer chaque point avec un cercle
        plt.text(x, y, nom)         # Ajouter le nom du point à côté du marqueur

    # Configuration du titre, des axes et de la grille de la figure.
    plt.title("Représentation graphique des points de référence d'un clou sur un plan cartésien")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()  # Affichage de la figure

# Fonction pour visualiser une image originale et son équivalent en niveaux de gris.
def visualiser_image_couleur_ng(chemin_vers_image_org, chemin_vers_image_ng):
    # Chargement des images
    image1 = Image.open(chemin_vers_image_org)
    image2 = Image.open(chemin_vers_image_ng)

    # Création d'une figure avec deux sous-graphiques pour les deux images.
    plt.figure(figsize=(10, 5))  # Taille de la figure

    # Configuration et affichage de chaque image dans son sous-graphique.
    plt.subplot(1, 2, 1)  # Premier sous-graphique pour l'image originale
    plt.imshow(image1)
    plt.title('Image Originale')

    plt.subplot(1, 2, 2)  # Deuxième sous-graphique pour l'image en niveaux de gris
    plt.imshow(image2)
    plt.title('Image en NG')

    plt.show()  # Affichage de la figure

# Fonction pour visualiser les transformations appliquées à une image en niveaux de gris.

def visualiser_transforms_image(path_image_ng, radius=2):
    """
    Visualise les transformations appliquées sur une image en niveaux de gris.

    Args:
    path_image_ng (str): Chemin de l'image en niveaux de gris.
    radius (int): Rayon utilisé dans la deuxième transformation.
    """

    # Ouverture de l'image et conversion en tableau NumPy pour le traitement
    img = Image.open(path_image_ng).convert('L')
    img_array = np.array(img)

    # Application des transformations sur l'image
    image_trasf_1 = appliquer_transformation_1(img_array)
    image_trasf_2 = appliquer_transformation_2(img_array, radius)

    # Préparation des images et de leurs titres pour l'affichage
    images = [img_array, image_trasf_1, image_trasf_2]
    titles = ['Image en NG', 'Image après transformation 1', 'Image après transformation 2']

    # Création d'une figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 ligne, 3 colonnes

    # Affichage de chaque image dans un sous-graphique
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])
        ax.axis('off')  # Désactivation des axes pour une meilleure visibilité

    plt.show()  # Affichage de la figure complète

def visualiser_seg_image(path_image_ng, radius=2, w=3):
    """
    Visualise l'image segmentée après avoir appliqué une transformation et calculé un histogramme.

    Args:
    path_image_ng (str): Chemin de l'image en niveaux de gris.
    radius (int): Rayon utilisé dans la transformation.
    w (int): Largeur de la fenêtre utilisée pour le calcul de l'histogramme.
    """

    # Ouverture et traitement de l'image
    img = Image.open(path_image_ng).convert('L')
    img_array = np.array(img)
    image_trasf_2 = appliquer_transformation_2(img_array, radius)

    # Calcul de l'histogramme et segmentation
    tab_histo = calculer_histogramme(image_trasf_2, w)
    labels2 = regrouper_points(tab_histo)
    segmented_image = labels2.reshape(img_array.shape[0] - w + 1, img_array.shape[1] - w + 1)

    # Affichage de l'image segmentée
    plt.imshow(segmented_image, cmap='gray')
    plt.title("Image Segmentée")
    plt.show()  # Affichage de l'image segmentée

def visualiser_transformations_clou(*coordonnees_lists):
    """
    Visualise les transformations appliquées sur les coordonnées d'un clou.

    Args:
    coordonnees_lists (tuple): Listes de coordonnées transformées à afficher.
    """

    # Création d'une figure avec plusieurs graphiques pour chaque transformation
    fig, axes = plt.subplots(len(coordonnees_lists), 1, figsize=(6, 12))

    # Titres pour chaque graphique
    titres = ["Réflexion", "Rotation", "Inclinaison"]

    # Affichage des points transformés sur chaque graphique
    for ax, coordonnees, titre in zip(axes, coordonnees_lists, titres):
        for nom, (x, y) in coordonnees:
            ax.plot(x, y, marker='o')
            ax.text(x, y, nom)

        ax.set_title(titre)
        ax.set_xlabel("X")
        ax.axis('equal')
        ax.set_ylabel("Y")
        ax.grid(True)

    # Ajustement de l'espacement pour éviter le chevauchement
    plt.tight_layout(pad=3.0)

    plt.show()  # Affichage de la figure complète

if __name__ == '__main__':
    # Bloc principal pour exécuter les fonctions de visualisation

    # Visualisation des points du clou, des transformations appliquées,
    # des images en couleur et en niveaux de gris, des transformations d'image
    # et de l'image segmentée.
    visualiser_points_clou(__COORDS_CLOU)
    visualiser_transformations_clou(__REFLECTED_COORD, __ROTATED_COORD, __INCLIN_COORD)
    visualiser_image_couleur_ng(__PATH_IMAGE_ORIG, __PATH_IMAGE_NG)
    visualiser_transforms_image(__PATH_IMAGE_NG)
    visualiser_seg_image(__PATH_IMAGE_NG)
