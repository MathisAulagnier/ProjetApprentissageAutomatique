"""
Pour ajouter une fonctionnalité de suivi du chemin parcouru dans l'arbre pendant la prédiction
"""

def predict_with_trace(self, X):
    """
    Prédit la classe pour de nouvelles instances et enregistre le chemin parcouru dans l'arbre de décision.
    X: Ensemble d'attributs à tester (Pandas DataFrame)
    Retourne les prédictions et le chemin parcouru dans l'arbre.
    """
    X = self.discretize(X, entrainement_en_cours=False)
    chemins = []  # Liste pour enregistrer le chemin parcouru pour chaque exemple
    predictions = []  # Liste pour enregistrer les prédictions pour chaque exemple

    for n_candidate in range(len(X)):
        node = self.tree
        chemin = []  # Liste qui va enregistrer le chemin pour chaque exemple

        while isinstance(node, dict):  # Tant que nous sommes sur un nœud interne
            attribut = list(node.keys())[0]  # Prendre l'attribut du nœud
            valeur = X[attribut].iloc[n_candidate]

            # Ajouter l'attribut et la valeur au chemin
            chemin.append(f"Attribut évalué: {attribut}, Valeur: {valeur}")

            # Si la valeur de l'attribut est dans les sous-nœuds
            if valeur in node[attribut]:
                node = node[attribut][valeur]
            else:
                chemin.append("Valeur inconnue, prédiction par défaut.")
                predictions.append(None)  # Valeur inconnue, on peut ajouter une valeur par défaut ici
                break
        else:
            # Le nœud final est une feuille avec la prédiction
            chemin.append(f"Classe prédite: {node}")
            predictions.append(node)

        chemins.append(chemin)

    return predictions, chemins
