# Open AI Tanzania Challenge

Voir [la description du challenge](https://competitions.codalab.org/competitions/20100).

## Données

Le jeu de données considérée contient 13 images dans le set d'entraînement
accompagnées par autant de fichiers `.geojson` décrivant les bâtiments qui y
figurent, et 7 images dans le set de test. Toutes ces images sont à très haute
définition (entre 0.04 et 0.08 mètres par pixel), les couches rasters associées
mesurant entre 42k x 17k pixels et 75k x 75k pixels.

Trois classes de bâtiments se distinguent sur les photos aériennes ainsi
proposées :
- les bâtiments terminés :
- les bâtiments en cours de construction ;
- les bâtiments à l'état de fondation.

## Résultat attendu

Le challenge nécessite de procéder à une segmentation de chaque instance de
bâtiment présente sur les images du jeu de données test. Il est ainsi
nécessaire d'aller plus loin que la segmentation sémantique, où chaque pixel
est classé dans l'une des *classes* du jeu de données : ici, chaque pixel doit
être classé comme appartenant à un *bâtiment* particulier.

Ainsi, pour chacune des 7 images du jeu de données test, il faut produire un
fichier `.csv` contenant les informations suivantes :

| building_id | conf_foundation | conf_unfinished | conf_completed | coords_geo | coords_pixel | 
|-------------|-----------------|-----------------|----------------|------------|--------------|
| 1 | 0.3 | 0.5 | 0.2 | "POLYGON ((-5.7226 39.3043, -5.7227 39.3048, ... , -5.722 39.3043))" | "POLYGON ((714 978, 892 1045, ... , 714 978))" |
| 2 | ... | ... | ... | ... | ... |

où `building_id` est l'identifiant du bâtiment prédit, `conf_foundation`,
`conf_unfinished` et `conf_completed` respectivement les scores attribués par
le modèle au trois classes de bâtiments, et `coords_geo` et `coords_pixel` les
géométries associées, respectivement en coordonnées géographique et en pixel
sur l'image.

## Chaîne de traitement

Ce challenge donne l'occasion de mettre à l'épreuve une chaîne de traitement de
données géospatiales. Plusieurs étapes sont ainsi identifiées :
- le tuilage des images en briques de taille raisonnable (`i.e.` compatible
  avec l'utilisation d'un réseau de neurones convolutifs), à l'aide des outils
  GDAL
- l'extraction des informations géographiques (coordonnées de l'image,
  projection), toujours avec GDAL
- l'entrée en base des objets géographiques (bâtiments du jeu de données
  d'entraînement), via la commande `ogr2ogr`
- l'extraction des objets géographiques (bâtiments) propres à chaque tuile, via
  une requête PostGIS
- l'entraînement d'un modèle supervisé visant la détection automatique des
  instances de bâtiments sur une image, à partir des images d'entraînement, le
  modèle [Mask-RCNN](https://github.com/matterport/Mask_RCNN), référence de
  l'état de l'art en matière de segmentation d'instance est ici utilisé
- la prédiction des bâtiments figurant sur les images de tests découpées en tuiles
- le regroupement des prédictions faites dans l'ensemble des tuiles composant
  une même image, et ce pour l'ensemble des images du jeu de données test
- la gestion des doublons (bâtiments "coupés en deux" par le tuilage, et
  potentiellement détectée plus d'une fois)

L'ensemble de ces étapes est ici organisé autour d'un pipeline `Luigi`.

## Dépendances

Les dépendances du projet sont les suivantes :
- GDAL
- keras
- luigi
- Mask-RCNN
- numpy
- openCV
- pandas
- pillow

## Installation

L'installation du projet s'appuie classiquement sur l'utilisation d'un environnement virtuel :

```
git clone ssh://git@git.oslandia.net:10022/Oslandia-data/tanzania_challenge.git
cd tanzania_challenge
mkvirtualenv tanz
python setup.py install
```
