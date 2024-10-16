from copy import deepcopy

import networkx as nx
import matplotlib.pyplot as plt
import copy

class DecisionTreeVisualizer:
    def __init__(self, tree):
        """
        Initialisation
        :param tree: arbre de décision qu'on veut afficher
        """
        self.tree = deepcopy(tree)
        self.i=0 # Compteur pour les noeuds du graphique

    def draw_tree(self, level, x, graph, pos, parent_name, width):
        """
        Fonction récursive pour construire la représentation visuelle (graph) de l'arbre à partir du noeud self
        :param level: Niveau de profondeur
        :param x: Coordonnées horizontales d'un noeud
        :param graph: Graphique final
        :param pos: Dictionnaire contenant chaque noeud ainsi que sa position dans le graphique
        :param parent_name: Nom du noeud parent
        :param width: Largeur d'affichage
        """
        # Vérifie si c'est bien un dictionnaire
        if not isinstance(self.tree, dict):
            print("self.tree n'est pas un dictionnaire.")
            return

        num_children = len(self.tree)
        dx = width / max(num_children, 1) # Distance en coordonnées x entre chaque noeud
        next_x = x - width / 2 + dx / 2 # Position en x du prochain noeud

        for key, value in self.tree.items():
            node_name = f"{key}"

            #Ajouter le noeud avec le parent, en ajoutant un numéro unique à chaque noeud pour éviter les regroupements
            new_node_name = self.add_node(node_name, parent_name, next_x, level, graph, pos)

            if isinstance(value, dict):  # récursion pour placer les enfants
                self.tree = value
                self.draw_tree(level=level - 1, x=next_x, graph=graph,
                               pos=pos, parent_name=new_node_name, width=dx)
            else:
                #si c'est une feuille de l'arbre
                leaf_name = f"{value}"
                self.add_node(leaf_name, new_node_name, next_x, level-1, graph, pos)
            next_x += dx


    def add_node(self, leaf_name, parent_name, x_pos, y_pos, graph, pos):
        """
        Fonction permettant d'ajouter un noeud au graph
        :param leaf_name: Nom du noeud à ajouter
        :param parent_name: Nom du noeud parent
        :param x_pos: Position en x du noeud
        :param y_pos: Position en y du noeud
        :param graph: Graphique final
        :param pos: Dictionnaire contenant chaque noeud ainsi que sa position dans le graphique
        :return: Nom du noeud au graph + i (numéro du noeud) pour éviter les regroupements
        """
        leaf_name = leaf_name + "\n" + str(self.i)
        pos[leaf_name] = (x_pos, y_pos)
        graph.add_node(leaf_name)
        graph.add_edge(leaf_name, parent_name)
        self.i+=1
        return leaf_name


    def show_tree_graph(self):
        """
        Affiche l'arbre sous forme de graphe.
        """
        graph = nx.DiGraph()
        pos = {}
        graph.add_node("root")
        pos["root"] = (0, 0)
        self.draw_tree(-1, 0, graph, pos, "root", 1)
        # print("graph: ", graph)
        # print("pos", pos)

        # Dessiner le graphe avec networkx et matplotlib
        plt.figure(figsize=(50,15))
        nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold',
                arrows=False)
        plt.show()
