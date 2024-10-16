# TP1 - Fondamentaux de l'apprentissage automatique 8INF867

Ce projet implémente l'algorithme ID3 pour la construction d'arbres de décision. Il inclut également des tests unitaires pour vérifier le bon fonctionnement de l'implémentation.

## Membres du groupe projet

- Mathis AULAGNIER - _AULM12040200_
- Héléna BARBILLON - _BARH30530200_
- Moise KUMAVI - _KUMM06100200_



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
├── DecisionTreeVisulizer.py # Pour afficher l'arbre graphiquement
├── main.py
└── tests
    ├── __init__.py
    ├── ...
    └── tests_init.py
```

### Détails des Fichiers

- **id3.py** : Ce fichier contient l'implémentation de l'algorithme ID3 pour la construction d'arbres de décision.

- **DecisionTreeVisualizer.py** : Ce fichier contient un classe permettant d'afficher l'arbre dans un graphique après sa construction.

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


