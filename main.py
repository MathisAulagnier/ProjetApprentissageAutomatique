import numpy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ID3 import ID3
from DecisionTreeVisualizer import DecisionTreeVisualizer

nom_colonne_classe = "variety"

# Test de la classe ID3
arbre = ID3(depth_limit=None, nom_colonne_classe=nom_colonne_classe)

# Chargement des données
df = pd.read_csv('Data/iris.csv')
X = df.drop(nom_colonne_classe, axis=1)
y = df[nom_colonne_classe]

# Séparation des données en ensemble d'entraînement et ensemble de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Affichage des premières lignes du DataFrame
print(df.head(5))

print('\n\n')

# Entraînement de l'arbre avec l'ensemble d'entraînement
print("Entraînement de l'arbre...")
train_data = X_train.copy()
train_data[nom_colonne_classe] = y_train
arbre.fit(train_data)
print("Entraînement terminé.")

print('\n\n')

# Calcul des prédictions et affichage des premières prédictions
predictions, chemins = arbre.predict(X_train)
print("Prédictions (train): ", predictions[:5])

# Calcul de l'accuracy sur l'ensemble d'entraînement
accuracy_train = np.mean(predictions == y_train)
print("Accuracy (train): ", accuracy_train)

print('\n\n')

# Affichage de l'arbre avant élagage
print("Arbre avant élagage:\n")
arbre.print_tree()

# Application de l'élagage postérieur avec l'ensemble de validation
arbre.prune(X_val, y_val)

# Affichage de l'arbre après élagage
print("\n\nArbre après élagage:\n")
arbre.print_tree()

# Visualisation de l'arbre après élagage
visualizer = DecisionTreeVisualizer(arbre.tree)
visualizer.show_tree_graph()

# Prédictions et accuracy sur l'ensemble de validation après élagage
predictions_val, chemins_val = arbre.predict(X_val)
accuracy_val = np.mean(predictions_val == y_val)
print("\n\nAccuracy (validation) après élagage: ", accuracy_val)


