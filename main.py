import pandas as pd
import numpy as np
from id3 import ID3


# Test de la classe ID3
arbre = ID3(depth_limit=3,nom_colonne_classe="variety")
#arbre = ID3( nom_colonne_classe="admission", seuil_gini=0.0001, seuil_discretisation=10)
arbre.show_tree()

df = pd.read_csv('Data/iris.csv')
X = df.drop('variety', axis=1)
y = df['variety']


print(df.head(5))

print('\n\n')

print("Entraînement de l'arbre...")
arbre.fit(df)
print("Entraînement terminé.")

print('\n\n')

# Calculer l'accuracy
predictions, chemins = arbre.predict(X)

print("Prédictions: ", predictions[0:5])


if y.dtype == 'object':
    y = y.fillna('Etiquette manquante')
else:
    y = y.fillna(y.max() + 1)

accuracy = np.mean(predictions == y)
print("Accuracy: ", accuracy)

print('\n\n')

#print("Affichage de l'arbre:")
#arbre.show_tree()
