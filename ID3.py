import pandas as pd
import numpy as np
import copy  # Pour deepcopy


from sklearn.preprocessing import (
    KBinsDiscretizer, 
    FunctionTransformer
)
from sklearn.pipeline import (
    Pipeline
)

from sklearn.compose import (
    ColumnTransformer
)

from sklearn.impute import (
    SimpleImputer
)

from Predict_With_Trace import predict_with_trace


class ID3:
    def __init__(self, depth_limit=10, nom_colonne_classe='admission', seuil_gini=0.05, seuil_discretisation=10fit , number_bins=2):
        """
        Initialise l'arbre avec une limite de profondeur facultative.
        """
        self.depth_limit = depth_limit
        self.nom_colonne_classe = nom_colonne_classe
        self.seuil_gini = seuil_gini
        self.seuil_discretisation = seuil_discretisation
        self.moyennes = {}
        self.manquantes = {}
        self.model_discretisation = {}
        self.num_bins = number_bins
        self.tree = None
    
    def fit(self, df):
        """
        Fonction pour construire l'arbre de décision à partir des données d'apprentissage.
        X: Ensemble d'attributs (Pandas DataFrame)
        y: Classes cibles (Pandas Series ou array-like)
        """
        # Si Nom de la classe n'est pas présent dans le DataFrame, on renvoie une erreur
        if self.nom_colonne_classe not in df.columns:
            raise ValueError("ERREUR : La colonne candidate pour être le Label n'est pas présente dans ce DataFrame.")

        # Si le type de la colonne classe n'est pas object, on renvoie une erreur
        if df[self.nom_colonne_classe].dtype != 'object':
            raise ValueError("ERREUR : La colonne candidate pour être le Label n'est pas de type 'object'.")

        # X = df.drop(self.nom_colonne_classe, axis=1)
        # y = df[self.nom_colonne_classe]
        X = df.drop(self.nom_colonne_classe, axis=1).reset_index(drop=True)  # Réinitialise les index
        y = df[self.nom_colonne_classe].reset_index(drop=True)  # Réinitialise les index

        # Appelle la méthode récursive pour construire l'arbre
        X = self.discretize(X, entrainement_en_cours=True)

        # Convertir les NaN en 'Valeur manquante'
        if y.dtype == 'object':
            y = y.fillna('Etiquette manquante')
        else:
            y = y.fillna(y.max() + 1)
        
        self.tree = self.build_tree(X, y, depth=0)


    def build_tree(self, X, y, depth):
        """
        Méthode interne récursive pour construire l'arbre ID3.
        """
        # Si toutes les instances appartiennent à la même classe
        # Si profondeur max atteinte, on retourne la classe majoritaire ou il n'y a plus d'attributs
        # Si le critère de Gini est en dessous du seuil, on arrête la construction
        if len(np.unique(y)) == 1 or depth == self.depth_limit or len(X.columns) == 0 or self.gini(y) < self.seuil_gini:
            return y.mode().iloc[0]  # Si toutes les classes sont les mêmes, on retourne cette classe
    
        # Cherche le meilleur candidat pour la séparation
        best_candidat = self.best_split(X, y)
        
        # Si aucun attribut ne donne de bon gain, on arrête
        if best_candidat is None:
            return y.mode().iloc[0]
        
        # création d'un noeud de l'arbre
        tree = {best_candidat: {}}
        
        # Séparer le dataframe en sous-ensembles dépendant des valeurs de l'attribut choisi
        for valeur in np.unique(X[best_candidat]):
            sous_ensemble = X[X[best_candidat] == valeur].reset_index(drop=True)
            sous_ensemble_classes = y[X[best_candidat] == valeur].reset_index(drop=True)
            
            # Si le sous-ensemble est vide, retourner la classe majoritaire
            if len(sous_ensemble_classes) == 0:
                tree[best_candidat][valeur] = y.mode().iloc[0]
            else:
                # Construire récursivement l'arbre pour les sous-ensembles non vides
                tree[best_candidat][valeur] = self.build_tree(sous_ensemble.drop(columns=[best_candidat]), sous_ensemble_classes, depth+1)
        
        return tree
    def print_tree(self):
        tree_copy = self.copy()
        tree_copy.print_tree_recursion()

    def print_tree_recursion(self, level=0):
        """
        Affiche l'arbre construit.
        """
        # Vérifie si c'est bien un dictionnaire
        if not isinstance(self.tree, dict):
            #print("self.tree n'est pas un dictionnaire.")
            return
        # Si l'arbre est nul, on arrête la fonction
        if self.tree is None:
            #print("L'arbre n'a pas été construit.")
            return

        #print("len: ", len(self.tree))
        items = self.tree.items()
        for key, value in items:
            #print('\t'*level , key, "(key)")
            if isinstance(value, dict) and len(value)!=0: #si c'est une branche
                self.tree = value
                self.print_tree_recursion(level=level+1)
            else:
                pass
                #print('\t'*(level+1), value, "(value)") # si c'est une feuille de l'arbre

    def predict(self, X):
        """
        Prédit la classe pour de nouvelles instances à partir de l'arbre construit.
        X: Ensemble d'attributs à tester (Pandas DataFrame)
        Retourne un array-like des prédictions pour chaque instance.
        """
        X = self.discretize(X, entrainement_en_cours=False)
        chemin = []
        predictions = []
        for n_candidate in range(len(X)):
            node = self.tree
            while isinstance(node, dict):
                attribut = list(node.keys())[0]
                valeur = X[attribut].iloc[n_candidate]
                if valeur in node[attribut]:
                    chemin.append(node)
                    node = node[attribut][valeur]
                else:
                    chemin.append(node)
                    predictions.append(None)
                    #print("Valeur inconnue, on peut ajouter une valeur par défaut ici")
                    break
            else:
                chemin.append(node)
                predictions.append(node)
        return predictions, chemin
    
    def discretize(self, X, entrainement_en_cours=True):
        """
        Discrétise les attributs numériques si nécessaire via une pipeline scikit-learn.
        X: DataFrame des attributs
        Retourne une version discrétisée de X.
        """
        X = X.copy()
        
        # Sépare les colonnes selon leur type
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        boolean_cols = X.select_dtypes(include=['bool']).columns
        
        # Convertir les colonnes booléennes en entiers
        X[boolean_cols] = X[boolean_cols].astype(int)

        if entrainement_en_cours:
            # Crée des pipelines pour chaque type de donnée
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),  # Imputation des valeurs manquantes par la moyenne ## Réfléchir à une autre stratégie a passer en option
                ('discretizer', KBinsDiscretizer(n_bins=self.num_bins , encode='ordinal', strategy='uniform', subsample=None))  # Discrétisation # Proposer en option de choisir le nombre de bins et la stratégie
            ])

            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Objet manquant')),  # Imputation constante 
            ])

            boolean_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent'))  # Imputation du mode
            ])
            
            # Combine les transformations en un seul pipeline global
            self.preprocessor = ColumnTransformer([
                ('num', numeric_pipeline, numeric_cols),
                ('cat', categorical_pipeline, categorical_cols),
                ('bool', boolean_pipeline, boolean_cols)
            ])
        
            X_transformed = self.preprocessor.fit_transform(X)
            
        else:
            X_transformed = self.preprocessor.transform(X)
        
        # Convertir le résultat en DataFrame
        X_transformed = pd.DataFrame(X_transformed, columns=numeric_cols.tolist() + categorical_cols.tolist() + boolean_cols.tolist())
        
        return X_transformed


    def prune(self, X_val, y_val):
        """
        Applique un élagage postérieur (post-pruning) sur l'arbre.
        """
        X_val = self.discretize(X_val, entrainement_en_cours=False)

        def recursive_prune(node, X_val, y_val):
            # Si on est à une feuille, on retourne la classe actuelle
            if not isinstance(node, dict):
                return node

            # Sinon, nous sommes à un noeud interne
            attribut = list(node.keys())[0]
            sous_noeuds = node[attribut]

            # Réinitialisation des index de X_val et y_val pour les aligner
            X_val_reset = X_val.reset_index(drop=True)
            y_val_reset = y_val.reset_index(drop=True)

            # Parcourir les sous-arbres pour les élaguer
            for valeur, sous_arbre in sous_noeuds.items():
                indices = X_val_reset[attribut] == valeur

                X_val_subset = X_val_reset.loc[indices]
                y_val_subset = y_val_reset.loc[indices]

                if len(X_val_subset) > 0:
                    sous_noeuds[valeur] = recursive_prune(
                        sous_arbre,
                        X_val_subset,
                        y_val_subset
                    )

            # Après avoir parcouru les enfants, on verifie si l'on doit élaguer ce noeud
            predictions = []
            for valeur, sous_arbre in sous_noeuds.items():
                if isinstance(sous_arbre, dict):
                    return node  # Garder le noeud si un enfant est encore un sous-arbre
                predictions.append(sous_arbre)

            # Si toutes les feuilles prédisent la même classe, on peut élaguer
            if len(set(predictions)) == 1:
                return predictions[0]

            return node

        # Commencer l'élagage depuis la racine
        self.tree = recursive_prune(self.tree, X_val, y_val)

    def gini(self, y):
        """
        Calcule l'entropie de l'ensemble de données y.
        y: Classes cibles
        Retourne le coefficiant de Gini.
        """
        valeurs, occurrences = np.unique(y, return_counts=True)
        gini = 1
        for occ in occurrences:
            gini -= (occ / len(y))**2
        return gini

    def information_gain(self, X_column, y):
        """
        Calcule le gain d'information pour un attribut donné.
        X_column: Une colonne du DataFrame (attribut)
        y: Classes cibles
        Retourne le gain d'information.
        """
        gini_parent = self.gini(y)
        gini_enfant = 0.0

        valeurs, occurrences = np.unique(X_column, return_counts=True)

        for valeur, occ in zip(valeurs, occurrences):
            sous_ensemble = y[X_column == valeur]
            gini_enfant += (occ / len(X_column)) * self.gini(sous_ensemble)
        
        gain = gini_parent - gini_enfant
        return gain
    


    def best_split(self, X, y):
        """
        Trouve le meilleur attribut sur lequel séparer les données.
        X: Ensemble d'attributs
        y: Classes cibles
        Retourne le meilleur attribut et les ensembles séparés.
        """
        principaux_candidats = None
        gain_max = -1
        for i, candidat in enumerate(X.columns):
            gain = self.information_gain(X[candidat], y)
            if gain > gain_max:
                principaux_candidats = candidat
                gain_max = gain
        return principaux_candidats
    

    def copy(self):
        """
        Copie de l'arbre
        """
        copied_tree = copy.deepcopy(self)
        return copied_tree
