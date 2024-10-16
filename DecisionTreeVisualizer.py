from copy import deepcopy

import networkx as nx
import matplotlib.pyplot as plt
import copy

class DecisionTreeVisualizer:
    def __init__(self, tree):
        self.tree = deepcopy(tree)

    def draw_tree(self, level, x, graph, pos, parent_name, width):

        # Vérifie si c'est bien un dictionnaire
        if not isinstance(self.tree, dict):
            print("self.tree n'est pas un dictionnaire.")
            return
        # Si l'arbre est nul, on arrête la fonction
        if self.tree is None:
            print("L'arbre n'a pas été construit.")
            return

        num_children = len(self.tree)
        dx = width / max(num_children, 1)
        next_x = x - width / 2 + dx / 2

        items = self.tree.items()
        for key, value in items:
            # Ajouter le niveau au nom du nœud pour le rendre unique
            node_name = f"{key}\n{level}"

            #Ajouter le noeuf avec le parent
            self.add_node(node_name, parent_name, next_x, level, graph, pos)

            if isinstance(value, dict):  # récursion pour placer les enfants
                self.tree = value
                self.draw_tree(level=level - 1, x=next_x, graph=graph,
                               pos=pos, parent_name=node_name, width=dx)
            else:
                #si c'est une feuille de l'arbre
                leaf_name = f"{value}\n{level - 1}"
                self.add_node(leaf_name, node_name, next_x, level-1, graph, pos)
            next_x += dx


    def add_node(self, leaf_name, parent_name, x_pos, y_pos, graph, pos):
        pos[leaf_name] = (x_pos, y_pos)
        graph.add_node(leaf_name)
        graph.add_edge(leaf_name, parent_name)


    def show_tree_graph(self):
        """
        Affiche l'arbre sous forme de graphe.
        """

        graph = nx.DiGraph()
        pos = {}
        graph.add_node("root")
        pos["root"] = (0, 0)
        # Créer le graphe et obtenir les positions des nœuds
        # graph, pos = self.draw_tree()
        self.draw_tree(-1, 0, graph, pos, "root", 1)
        print("graph: ", graph)
        print("pos", pos)

        # Dessiner le graphe avec networkx et matplotlib
        plt.figure(figsize=(30,15))
        nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold',
                arrows=False)
        plt.show()
