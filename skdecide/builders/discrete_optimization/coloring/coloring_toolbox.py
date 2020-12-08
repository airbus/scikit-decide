import networkx as nx


def compute_cliques(g, nb_max=None):
    cliques = []
    nb = 0
    for c in nx.algorithms.clique.find_cliques(g):
        print(c)
        cliques += [c]
        nb += 1
        if nb_max is not None and nb >= nb_max:
            not_all = True
            break
    return cliques, not_all

