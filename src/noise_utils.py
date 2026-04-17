import numpy as np

def remove_edges(G, ratio=0.1):
    G = G.copy()
    edges = list(G.edges())
    remove_num = int(len(edges) * ratio)
    if remove_num == 0:
        return G

    idxs = np.random.choice(len(edges), remove_num, replace=False)
    for idx in idxs:
        u, v = edges[idx]
        if G.has_edge(u, v):
            G.remove_edge(u, v)
    return G


def mask_attributes(X, ratio=0.1):
    X = X.copy()
    mask = np.random.rand(*X.shape) < ratio
    X[mask] = 0.0
    return X


def build_noisy_pair(Gs, Gt, Xs, Xt, edge_noise=0.0, attr_noise=0.0):
    Gs_n = remove_edges(Gs, edge_noise)
    Gt_n = remove_edges(Gt, edge_noise)

    Xs_n = mask_attributes(Xs, attr_noise)
    Xt_n = mask_attributes(Xt, attr_noise)
    return Gs_n, Gt_n, Xs_n, Xt_n