# Classe ID3 pour Arbre de Décision

## Description

Ce fichier README explique l'implémentation et l'utilisation de la classe ID3, un classifieur d'arbre de décision personnalisé pour les données catégorielles.
Prérequis

1. Python 3.x
2. Packages : pandas, numpy, scikit-learn

## Installation

Pour installer les dépendances requises, exécutez :

```sh

pip install pandas numpy scikit-learn
```

Alternativement, créez et activez un environnement virtuel pour gérer les dépendances :

```sh

# Créer un environnement virtuel
python -m venv env

# Activer l'environnement virtuel (Windows)
env\Scripts\activate

# Activer l'environnement virtuel (macOS/Linux)
source env/bin/activate

# Installer les dépendances
pip install pandas numpy scikit-learn
```

Initialisation de la Classe

```python

from id3 import ID3

# Exemple d'initialisation
model = ID3(depth_limit=5, nom_colonne_classe='admission', seuil_gini=0.05, seuil_discretisation=10, number_bins=10)
```
Paramètres
- depth_limit : Profondeur maximale de l'arbre.
- nom_colonne_classe : Nom de la colonne cible dans le dataset.
- seuil_gini : Seuil d'impureté de Gini pour la division des nœuds.
- seuil_discretisation : Seuil pour la discrétisation des données.
- number_bins : Nombre de bins pour discrétiser les caractéristiques continues.

Méthodes

1. fit(df)

Construit l'arbre de décision à partir des données d'apprentissage.
Paramètres :
    df : DataFrame Pandas contenant les données d'apprentissage, y compris la colonne cible.
Utilisation :
```python
model.fit(training_data)
predict(X)
```

Prédit les étiquettes de classe pour de nouvelles instances en utilisant l'arbre construit.
Paramètres : X : DataFrame Pandas contenant les données de test.
Retourne :predictions : Liste des étiquettes de classe prédites.
chemin : Liste des chemins de décision.
Utilisation :
```python
predictions, chemin = model.predict(test_data)
```

## Exemple d'Utilisation

```python

import pandas as pd
from sklearn.model_selection import train_test_split

# Charger vos données dans un DataFrame Pandas
df = pd.read_csv('Data/MBA.csv')

# Séparer les données en ensembles d'entraînement et de test (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialiser le modèle ID3
model = ID3(depth_limit=5, nom_colonne_classe='admission', seuil_gini=0.05, seuil_discretisation=10, number_bins=10)

# Entraîner le modèle sur les données d'apprentissage
model.fit(train_df)

# Faire des prédictions sur le dataset de test
predictions, chemin = model.predict(test_df.drop(columns=['admission']))

```

## Notes

- Assurez-vous que la colonne cible est de type object dans votre DataFrame.
- La méthode predict retourne un tuple contenant les prédictions et les chemins de décision.
- La méthode prune est un placeholder et n'est actuellement pas implémentée.

## Conclusion

La classe ID3 offre une implémentation personnalisable de l'algorithme d'arbre de décision ID3, capable de gérer les valeurs manquantes et de discrétiser les caractéristiques continues.