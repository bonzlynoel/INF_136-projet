# Importation des bibliothèques nécessaires
from visualisation import *
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from math import sin,cos,pi,tan

# Press the green button in the gutter to run the script.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# Cette fonction applique une réflexion miroir à un point par rapport à un axe spécifié  

# Arguments:
    # point (tuple) : représent les coordonnées du points à réfléchir 
    # axe (str) : représent l'axe de réflexion 'x' ou 'y'. Si l'axe choisie est x, il y aura réflexion par rapport à l'axe verticale (miroir horizontal), : changement de la coordonnées y
    # si l'axe 'y' est choisie il y aura reflexion par rapport à l'axe horizontal(miroir vertical) : changement de la coordonnées x   
# Retourne :
    # (tuple) : Représent les nouvelles coordonnées du point refletté 

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


# Cette fonction fait tourner un point dans le plan cartésien, autour d'un point centrale donné d'un certain angle (!!attention la varaition du centre n'a pas été testé mais à été pris en considération)

# Arguments :
    # point (tuple) : représent les coordonnées du points à faire pivoter 
    # angle (float) : représent l'angle de rotation en degrés. une valeur positve provoque une rotation antihoraire une valeur negative provoque une rotation horaire 
    # center( tuple) : représente les coordonnées du centre de rotation (par défaut: (0,0))
# Retourne :
    # (tuple) : Représent les nouvelles coordonnées du point après rotation. (arrondie à deux chiffres après la virgule) 

def calculer_rotate_point(point,angle,center):
        rad = angle * (pi / 180)

        point_liste = [point[0] - center[0], point[1] - center[1]]

        point_liste[0] = point_liste[0] * cos(rad) + point_liste[1] * (-sin(rad))
        #print(f'{point_liste[0]}')
        point_liste[1] = (point[0] - center[0]) * sin(rad) + point_liste[1] * cos(rad)  # attention au centre
        #print(f'{point_liste[1]}')
        return round(point_liste[0] + center[0],2), round(point_liste[1] + center[1],2)# round permet d'arrondir la valeur


# Cette fonction applique une inclinaison (cisaillement) sur un point. L'inclinaison 
# est déterminée par un angle qui peut être en 'x' ou en 'y'.

# Arguments:
    # point (tuple): représente les coordonnées du point à incliner (x,y) 
    # angle (float): représente l'intensité de l'angle d'inclinaison en degrés.
    # direction (str): représent la direction de l'inclinaison, pour un inclinaison horizontale utliser 'x' et pour un inclinaison verticale utiliser 'y'

# Retourne:
    # tuple : valeur de coordonnée après inclinaison. (Arrondie à deux chiffres après la virgule)

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


# Section 9 : 
    # Calcule et transformation géométrique des coordonnées d'un Clou


# Cette fonction détermine les coordonnées d'un clou en suivant la paramétrisation dans la Figure 1

# Arguments : 
    # a,b,c,d,e (float) : représente les dimensions spécifiques du clou, utilisé pour calculer les coordonnées.
# Retourne : 
    # list : Une liste de tuple de laquelle chaque tuples contient : une Chaine de caractères ( nom du point) et un tuple de deux nombres (float, float)
        représentant les coordonnées du point dans un plan 2D

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
    
# Cette fonction prend un ensemble de points clés représentatn un clou et applique trois types de transformation géométrique.(rotation,réflexion,inclinaision)
# chaque transformation est appliquée séquentiellement à tous les points clés.

# Arguments : 
    # points_clou (list) : Une liste de tuple de laquelle chaque tuples contient : une Chaine de caractères ( nom du point) et un tuple de deux nombres (float, float)
    # représentant les coordonnées du point dans un plan 2D
    # center_rotation( tuple) : représente les coordonnées du centre de rotation 
    # angle_rotation (float) : l'angle de rotation en degrés
    # angle_inclinaison (float) : l'angle d'inclinaison en degrés
    # direction_inclinaison (float) : la direction de l'inclinaison ('x' ou 'y')
    # axe_reflexion (str) : l'axe de reflexion ('x' ou 'y')
# Retourne : 
    # tuple : trois listes de tuple de lesquelles chaque tuples contient : une Chaine de caractères ( nom du point) et un tuple de deux nombres (float, float)
    # représentant les coordonnées du point dans un plan 2D après chaque transformation.

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


#Section 10 :
    #Extraction de caractéristiques à partir de données visuelles


#Cette fonction transforme une image en couleur en une nouvelle image en niveaux de gris.

#Arguments :
    #path_image_orig (str): Le chemin de l'image en couleur à transformer.
    #path_image_ng (str): Le chemin où sauvegarder l'image résultante en niveaux de gris.
#Retourne :
    #rien
###commentaire :
    #on doit définir le chemin pour se rendre à l'image de couleur dans 'chemin_image_couleur' et le chemin de l'image grise dans 'chemin_image_gris'

def appliquer_rgb_to_gry(path_image_orig, path_image_ng):
    
    #Charger l'image en couleur
    image_couleur = Image.open(path_image_orig)

    #Convertir l'image en niveaux de gris
    image_array = convertir_rgb_to_gry(image_couleur)

    # Sauvegarder l'image en niveaux de gris
    image_array.save(path_image_ng)

def convertir_rgb_to_gry(image_couleur):
    
    largeur, hauteur = image_couleur.size
    image_array = Image.new('L', (largeur, hauteur))

    for y in range(hauteur):
        for x in range(largeur):
            # Obtenir les composantes RGB du pixel
            r, g, b = image_couleur.getpixel((x, y))

            # Calculer la moyenne des composantes RGB
            moyenne = round((r + g + b) / 3)

            # Définir la nouvelle valeur de pixel en niveaux de gris
            image_array.putpixel((x, y), moyenne)

    return image_array
####Test à faire

#Cette fonction prend une image en niveaux de gris sous forme d'un tableau NumPy 2D et applique une transformation pour simplifier et
#extraire des caractéristiques significatives de l'image.

#Arguments :
    #image_gris (numpy.ndarray): Un tableau 2D NumPy représentant une image en niveaux de gris. Chaque élément du tableau correspond à
    #l'intensité d'un pixel de l'image.
#Retourne :
    #numpy.ndarray: Un tableau 2D NumPy résultant de la transformation appliquée.

def appliquer_transformation_1(image_array):
    
    #Créer un tableau pour stocker les résultats de la transformation
    resultat_transformation = np.zeros_like(image_array, dtype=np.uint8)

    #Dimensions de l'image
    hauteur, largeur = image_array.shape

    #Parcourir chaque pixel de l'image
    for y in range(1, hauteur - 1):  # Ignorer les bords pour éviter les problèmes
        for x in range(1, largeur - 1):
            #Obtenir les valeurs de gris des voisins
            voisins = [
                image_array[y - 1, x - 1], image_array[y - 1, x], image_array[y - 1, x + 1],
                image_array[y, x - 1], image_array[y, x + 1],
                image_array[y + 1, x - 1], image_array[y + 1, x], image_array[y + 1, x + 1]
            ]
            #Pixel central
            pixel_central = image_array[y, x]

            #Comparaison avec les voisins pour former le motif binaire
            motif_binaire = ''.join(['1' if voisin >= pixel_central else '0' for voisin in voisins])

            #Convertir le motif binaire en valeur décimale
            valeur_decimale = int(motif_binaire, 2)

            #Assigner la valeur décimale au pixel correspondant dans le résultat de la transformation
            resultat_transformation[y, x] = valeur_decimale

    return image_trasf_1
#### Test à faire

# Cette fonction transforme les données visuelles complexes d’une image en ensembles de caractéristiques plus simples et plus significatives.

# Arguments image_gris (numpy.ndarray):
    # Un tableau 2D NumPy représentant une image en niveaux de gris.
    # rayon (int): Un entier spécifiant le rayon du voisinage à considérer pour chaque pixel lors de la transformation.
# Retourne numpy.ndarray:
    # Un tableau 2D NumPy résultant de la transformation appliquée. 
    # Cette transformation est basée sur le rayon spécifié et peut modifier les caractéristiques visuelles originales de l'image.

def appliquer_transformation_2(image_array, radius):
    # Créer un tableau pour stocker les résultats de la transformation
    resultat_transformation = np.zeros_like(image_array, dtype=np.float32)

    # Dimensions de l'image
    hauteur, largeur = image_array.shape

    # Parcourir chaque pixel de l'image
    for y in range(rayon, hauteur - radius):  # Ignorer les bords pour éviter les problèmes
        for x in range(rayon, largeur - radius):
            # Calculer la valeur de sortie O(x,y) pour chaque pixel en utilisant la formule donnée
            valeur_sortie = (
                    np.log10(1 + abs(image_array[y, x + radius] - 2 * image_array[y, x] + image_array[y, x - radius])) +
                    np.log10(1 + abs(image_array[y + radius, x] - 2 * image_array[y, x] + image_array[y - radius, x])) +
                    np.log10(1 + abs(image_array[y - rayon, x + radius] - 2 * image_array[y, x] + image_array[y + radius, x - radius]))
            )

            # Assigner la valeur de sortie au pixel correspondant dans le résultat de la transformation
            resultat_transformation[y, x] = valeur_sortie

    # Remplacer les valeurs des pixels de bord par zéro
    resultat_transformation[:radius, :] = 0
    resultat_transformation[-radius:, :] = 0
    resultat_transformation[:, :radius] = 0
    resultat_transformation[:, -radius:] = 0

    # Convertir la matrice résultante de float à int
    resultat_transformation = resultat_transformation.astype(np.int)

    return image_trasf_2
#### Test à faire


#Section 11 :
    #Création et comparaison d'histogrammes


# Cette fonction génère un histogramme pour chaque pixel de l'image en utilisant un carré de voisinage de taille spécifiée.

# Arguments :
    # tableau_2D (numpy.ndarray): Un tableau 2D NumPy représentant une image.
    # w (int): La taille du carré de voisinage autour de chaque pixel pour lequel l'histogramme est calculé.
#Retourne :
    #numpy.ndarray: Un tableau 2D NumPy où chaque ligne représente un histogramme pour le carré correspondant de l'image.

def calculer_histogramme(image_trasf_2, w):
    
    # Déterminer la taille de l'image
    hauteur, largeur = image_trasf_2.shape

    # Calculer la valeur maximale dans le tableau 2D
    max_value = np.max(image_trasf_2)

    # Créer un tableau pour stocker les histogrammes
    tab_histo = np.zeros((hauteur - w + 1, largeur - w + 1, 5), dtype=np.int)
    # Le dernier axe (5) représente les bins: [0, max/4, max/2, 3*max/4, max]

    # Parcourir chaque fenêtre dans l'image
    for y in range(hauteur - w + 1):
        for x in range(largeur - w + 1):
            # Extraire la fenêtre
            fenetre = image_trasf_2[y:y + w, x:x + w]

            # Calculer l'histogramme de la fenêtre
            hist, _ = np.histogram(fenetre, bins=[0, max_value / 4, max_value / 2, (3 * max_value) / 4, max_value],
                                   range=(0, max_value))

            # Assigner l'histogramme au tableau des histogrammes
            tab_histo[y, x, :] = hist

    return tab_histo
#### Test à faire

# Cette fonction calcule la distance entre deux histogrammes.
# Arguments:
    # histogramme1 (numpy.ndarray): Premier histogramme sous forme de tableau 1D NumPy.
    # histogramme2 (numpy.ndarray): Deuxième histogramme sous forme de tableau 1D NumPy
# Retourne float:
    # La distance entre les deux histogrammes.

def calculer_distance_1(h1, h2):
    
    # Vérifier que les histogrammes ont la même taille
    if len(h1) != len(h2):
        print(ValueError("Les histogrammes doivent avoir la même taille."))

    # Calculer la distance entre les deux histogrammes
    distance_carree = np.sum((h1 - h2) ** 2)
    distance = np.sqrt(distance_carree)

    # Arrondir le résultat à deux chiffres après la virgule
    distance_arrondie_1 = round(distance, 2)

    return distance_arrondie_1
#### Test à faire

# Calculer la distance entre deux histogrammes.
# Arguments : 
    # histogramme1 (numpy.ndarray): Premier histogramme sous forme de tableau 1D NumPy.
    # histogramme2 (numpy.ndarray): Deuxième histogramme sous forme de tableau 1D NumPy
Retourne float:
    # La distance entre les deux histogrammes.

def calculer_distance_2(h1, h2):
    # Vérifier que les histogrammes ont la même taille
    if len(h1) != len(h2):
        print(ValueError("Les histogrammes doivent avoir la même taille."))

    # Calculer la distance entre les deux histogrammes
    distance = np.sum(np.abs(h1 - h2))

    # Arrondir le résultat à deux chiffres après la virgule
    distance_arrondie_2 = round(distance, 2)

    return distance_arrondie_2
#### Test à faire

# Section 12
    # Segmentation de données en groupes basée sur les histogrammes des points


# Cette fonction divise un ensemble de points dans un plan 2D en un nombre défini de groupes.

# Arguments data (numpy.ndarray):
    # Un tableau 2d numpy représentant l'ensemble de données à partitionner. Chaque ligne du tableau représente un histogramme décrivant un point.
    # k (int): Le nombre de groupes à identifier dans l'ensemble de données.
    # max_iterations (int): Le nombre maximal d'itérations que l'algorithme exécutera. La valeur par défaut est 50.
# Retourne numpy.ndarray:
    # Un tableau numpy 1D où chaque élément correspond à l'indice du centre le plus proche pour chaque point de l'ensemble de données. 
    # C'est un vecteur d'entiers de la même longueur que le nombre de points dans 'data', indiquant l'affectation de groupe pour chaque point.

def regrouper_points(data, k, max_iterations=50):
    
    # Initialiser les centres de groupes aléatoirement
    indices_aleatoires = np.random.choice(len(data), k, replace=False)
    centres = data[indices_aleatoires]

    # Répéter jusqu'à convergence ou jusqu'au nombre maximal d'itérations
    for _ in range(max_iterations):
        # Assigner chaque point au groupe le plus proche
        groupes = []
        for point in data:
            distances = [calcul_distance_1(point, centre) for centre in centres]
            groupe = np.argmin(distances)
            groupes.append(groupe)

        # Mettre à jour les centres de groupe
        nouveaux_centres = []
        for groupe in range(k):
            points_groupe = data[np.array(groupes) == groupe]
            centre_groupe = np.mean(points_groupe, axis=0)
            nouveaux_centres.append(centre_groupe)

        # Vérifier la convergence
        if np.allclose(centres, nouveaux_centres):
            break

        centres = np.array(nouveaux_centres)

    return np.array(groupes)
#### Test a faire

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
