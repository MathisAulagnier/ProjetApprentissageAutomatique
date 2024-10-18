import numpy as np

def prune(self, X_val, y_val):
    """
    Applique le post-élagage à l'arbre de décision.
    X_val: Ensemble d'attributs de validation df
    y_val: Classes cibles de validation
    """

    def _prune_node(node, X_val, y_val):
        """
        Fonction récursive qui élaguera le sous-arbre si cela améliore la performance sur l'ensemble de validation.
        """
        if not isinstance(node, dict):
            # Si le nœud est une feuille, on ne fait rien
            return node

        # Prendre le premier attribut de séparation dans le nœud actuel
        attribut = list(node.keys())[0]

        for valeur in node[attribut]:
            # Pour chaque valeur, on descend dans l'arbre
            if isinstance(node[attribut][valeur], dict):
                # Applique l'élagage récursivement sur le sous-arbre
                node[attribut][valeur] = _prune_node(node[attribut][valeur], X_val[X_val[attribut] == valeur],
                                                     y_val[X_val[attribut] == valeur])

        # Une fois les sous-arbres élagués, on vérifie si ce nœud peut être remplacé par une feuille
        sous_ensemble_val = X_val[attribut].map(lambda x: node[attribut].get(x, None)).dropna()
        if len(sous_ensemble_val) == 0:
            return y_val.mode().iloc[0]  # Si tous les sous-ensembles sont vides, on remplace par la classe majoritaire

        # Prédictions sans élagage (sur ce nœud)
        predictions_avant = [node[attribut].get(v, y_val.mode().iloc[0]) for v in X_val[attribut]]

        # Calculer la précision avant l'élagage
        precision_avant = np.mean([pred == vrai for pred, vrai in zip(predictions_avant, y_val)])

        # Si on remplace le nœud par une feuille (classe majoritaire)
        feuille_proposee = y_val.mode().iloc[0]
        predictions_apres = [feuille_proposee] * len(y_val)
        precision_apres = np.mean([pred == vrai for pred, vrai in zip(predictions_apres, y_val)])

        # Si la précision après élagage est meilleure ou égale, on élaguer ce sous-arbre
        if precision_apres >= precision_avant:
            return feuille_proposee
        else:
            return node  # Sinon, on garde le sous-arbre

    # Appel récursif sur tout l'arbre
    self.tree = _prune_node(self.tree, X_val, y_val)
