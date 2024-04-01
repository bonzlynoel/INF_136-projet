# Importation des bibliothèques nécessaires
from visualisation import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from math import sin,cos,pi,tan

# Press the green button in the gutter to run the script.

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

"""
Cette fonction applique une réflexion miroir à un point par rapport à un axe spécifié  

Arguments:
    point (tuple) : représent les coordonnées du points à réfléchir 
    axe (str) : représent l'axe de réflexion 'x' ou 'y'. Si l'axe choisie est x, il y aura réflexion par rapport à l'axe verticale (miroir horizontal), : changement de la coordonnées y
        si l'axe 'y' est choisie il y aura reflexion par rapport à l'axe horizontal(miroir vertical) : changement de la coordonnées x   
Retourne :
    (tuple) : Représent les nouvelles coordonnées du point refletté 
"""


def calculer_reflexion_point(point, axe):
    a = 0
    b = 0
    if axe == 'x':
        a = point[1]
        b = -(point[0])


    elif axe == 'y':
        a = -(point[1])
        b = point[0]
    else:
        print(f'axe invalide')

    return (a, b)  # TUPLE REFLETTÉ


"""
Cette fonction fait tourner un point dans le plan cartésien, autour d'un point centrale donné d'un certain angle (!!attention la varaition du centre n'a pas été testé mais à été pris en considération)

Arguments:
    point (tuple) : représent les coordonnées du points à faire pivoter 
    angle (float) : représent l'angle de rotation en degrés. une valeur positve provoque une rotation antihoraire 
        une valeur negative provoque une rotation horaire 
    center( tuple) : représente les coordonnées du centre de rotation (par défaut: (0,0))
Retourne :
    (tuple) : Représent les nouvelles coordonnées du point après rotation. (arrondie à deux chiffres après la virgule) 
"""
def calculer_rotate_point(point,angle,center):
        rad = angle * (pi / 180)

        point_liste = [point[0] - center[0], point[1] - center[1]]

        point_liste[0] = point_liste[0] * cos(rad) + point_liste[1] * (-sin(rad))
        #print(f'{point_liste[0]}')
        point_liste[1] = (point[0] - center[0]) * sin(rad) + point_liste[1] * cos(rad)  # attention au centre
        #print(f'{point_liste[1]}')
        return round(point_liste[0] + center[0],2), round(point_liste[1] + center[1],2)# round permet d'arrondir la valeur



'''
Cette fonction applique une inclinaison (cisaillement) sur un point. L'inclinaison 
est déterminée par un angle qui peut être en 'x' ou en 'y'.

Arguments:
    point (tuple): représente les coordonnées du point à incliner (x,y) 
    angle (float): représente l'intensité de l'angle d'inclinaison en degrés.
    direction (str): représent la direction de l'inclinaison, pour un inclinaison horizontale utliser 'x' et pour un inclinaison verticale utiliser 'y'

Retourne:
    tuple : valeur de coordonnée après inclinaison. (Arrondie à deux chiffres après la virgule)

'''

def calculer_inclinaison_point(point,angle,direction):
    x = float(point[0])
    y = float(point[1])
    rad = angle * (pi / 180)
    if direction == 'x':
        x+= y*tan(rad)
        #print(f'{tan(rad)}')

    elif direction == 'y':
        y+= x*tan(rad)
    else:
        print(f'axes invalide')

    return round(x,2),round(y,2)
'''


SECTION 2 : Calcule et transformation géométrique des coordonnées d'un Clou



Cette fonction détermine les coordonnées d'un clou en suivant la paramétrisation dans la Figure 1

Arguments : 
    a,b,c,d,e (float) : représente les dimensions spécifiques du clou, utilisé pour calculer les coordonnées.
Retourne : 
    list : Une liste de tuple de laquelle chaque tuples contient : une Chaine de caractères ( nom du point) et un tuple de deux nombres (float, float)
        représentant les coordonnées du point dans un plan 2D
'''

def calculer_coordonnees_clou(a,b,c,d,e):

    pt_0 = (-b)/2, c/2
    pt_1 = (-b)/2, (-c)/2
    pt_2 = (-b)/2 - d, (-a)/2
    #print(f'{(-b)/2 }')
    pt_3 = (-b)/2 - d, a/2
    pk_0 = b/2+e, 0
    pk_1 = b/2, (-c)/2
    pk_2 = b/2, c/2

    liste = [('pt_0',pt_0),('pt_1',pt_1),('pt_2',pt_2),('pt_3',pt_3),('pk_0',pk_0),('pk_1',pk_1),('pt_2',pk_2)]
    return liste
'''
Cette fonction prend un ensemble de points clés représentatn un clou et applique trois types de transformation géométrique.(rotation,réflexion,inclinaision)
chaque transformation est appliquée séquentiellement à tous les points clés.

Arguments : 
    points_clou (list) : Une liste de tuple de laquelle chaque tuples contient : une Chaine de caractères ( nom du point) et un tuple de deux nombres (float, float)
        représentant les coordonnées du point dans un plan 2D
    center_rotation( tuple) : représente les coordonnées du centre de rotation 
    angle_rotation (float) : l'angle de rotation en degrés
    angle_inclinaison (float) : l'angle d'inclinaison en degrés
    direction_inclinaison (float) : la direction de l'inclinaison ('x' ou 'y')
    axe_reflexion (str) : l'axe de reflexion ('x' ou 'y')
Retourne : 
    tuple : trois listes de tuple de lesquelles chaque tuples contient : une Chaine de caractères ( nom du point) et un tuple de deux nombres (float, float)
        représentant les coordonnées du point dans un plan 2D après chaque transformation.
'''

def appliquer_transformation_clou(points_clou,center_rotation,angle_rotation,angle_inclinaison,direction_inclinaison,axe_reflexion):
    reflexion_points = []
    rotation_points = []
    inclinaison_points = []

    points = points_clou[1:len(points_clou)-1:2]
    print(f'{points}')
    for i in range(len(points)):
            reflexion_points[i-1] = calculer_reflexion_point(points[i-1],axe_reflexion)# la virgule fait une accumulation de points
            rotation_points[i-1] = calculer_rotate_point(points[i-1], angle_rotation, center_rotation)
            inclinaison_points[i-1] = calculer_inclinaison_point(points[i-1], angle_inclinaison, direction_inclinaison)

    return reflexion_points,rotation_points,inclinaison_points








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
