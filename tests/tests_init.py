import unittest
import pandas as pd
import numpy as np
from ID3 import ID3  # Importation absolue

class TestID3(unittest.TestCase):

    def test_init_default(self):
        model = ID3()
        self.assertIsNone(model.depth_limit, "Default depth_limit should be None.")
        self.assertEqual(model.nom_colonne_classe, 'admission', "Default nom_colonne_classe should be 'admission'.")
        self.assertEqual(model.seuil_gini, 0.05, "Default seuil_gini should be 0.05.")
        self.assertEqual(model.seuil_discretisation, 10, "Default seuil_discretisation should be 10.")
        self.assertEqual(model.moyennes, {}, "Default moyennes should be an empty dictionary.")
        self.assertEqual(model.manquantes, {}, "Default manquantes should be an empty dictionary.")
        self.assertEqual(model.model_discretisation, {}, "Default model_discretisation should be an empty dictionary.")
        self.assertIsNone(model.tree, "Default tree should be None.")

    def test_init_custom(self):
        model = ID3(depth_limit=5, nom_colonne_classe='result', seuil_gini=0.1, seuil_discretisation=5)
        self.assertEqual(model.depth_limit, 5, "Custom depth_limit should be 5.")
        self.assertEqual(model.nom_colonne_classe, 'result', "Custom nom_colonne_classe should be 'result'.")
        self.assertEqual(model.seuil_gini, 0.1, "Custom seuil_gini should be 0.1.")
        self.assertEqual(model.seuil_discretisation, 5, "Custom seuil_discretisation should be 5.")
        self.assertEqual(model.moyennes, {}, "Default moyennes should be an empty dictionary.")
        self.assertEqual(model.manquantes, {}, "Default manquantes should be an empty dictionary.")
        self.assertEqual(model.model_discretisation, {}, "Default model_discretisation should be an empty dictionary.")
        self.assertIsNone(model.tree, "Default tree should be None.")

if __name__ == '__main__':
    unittest.main()