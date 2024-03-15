
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
    point =(2,4)
    axe='x'
    angle =5
    center=(0,0)
    direction = 'y'
    #Implémantation des tranformations géométriques
    print(f'reflexion :{ calculer_reflexion_point(point, axe)}')
    print(f'rotation : {calculer_rotate_point(point,angle,center)}')
    print(f'inclinaison : {calculer_inclinaison_point(point,angle,direction)}')
    # Calcul et transformation géométrique des coordonnées d'un Clou
    a = 3
    b = 10
    c = 1
    d = 0.75
    e = 2
    print(f'coordonnées du clou : {calculer_coordonnees_clou(a,b,c,d,e)}')

    points_clou = calculer_coordonnees_clou(a,b,c,d,e)
    center_rotation =
    angle_rotation =
    angle_inclinaison =
    direction_inclinaison =
    axe_reflexion =

    print(f'coordonnées transformer :{appliquer_transformation_clou(points_clou,center_rotation,angle_rotation,angle_inclinaison,direction_inclinaison,axe_reflexion)} ')