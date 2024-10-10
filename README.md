# Projet d'Apprentissage Automatique

Ce projet implémente l'algorithme ID3 pour la construction d'arbres de décision. Il inclut également des tests unitaires pour vérifier le bon fonctionnement de l'implémentation.

## Structure du Workspace

```sh
.
├── Data # Dossier contenant les données utilisées pour les tests
│   ├── MBA.csv
│   ├── iris.csv
│   └── titanic.csv
├── README.md
├── Travail Pratique.pdf #Sujet du projet
├── __init__.py
├── id3.py # Implémentation de l'algorithme ID3
├── main.py
└── tests
    ├── __init__.py
    ├── ...
    └── tests_init.py
```

### Détails des Fichiers

- **id3.py** : Ce fichier contient l'implémentation de l'algorithme ID3 pour la construction d'arbres de décision.

- **tests/** : Ce répertoire contient les tests unitaires pour vérifier le bon fonctionnement de l'algorithme ID3.

- **data/** : Ce répertoire contient les jeux de données utilisés pour entraîner et tester l'algorithme ID3.

- **requirements.txt** : Liste des dépendances Python nécessaires pour exécuter le projet.

## Installation

Pour installer les dépendances nécessaires, exécutez la commande suivante :

```sh
pip install -r requirements.txt
```

## Exécution des Tests
Pour exécuter les tests unitaires, utilisez la commande suivante :
```sh
python -m unittest discover -s tests
```
