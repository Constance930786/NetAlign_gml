import numpy as np
import networkx as nx

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def load_base_graph(name="karate"):
    if name == "karate":
        G = nx.karate_club_graph()
    else:
        raise NotImplementedError("Add your benchmark graph loader here.")
    return G

def generate_node_features(G, feat_dim=16, mode="random"):
    n = G.number_of_nodes()
    if mode == "random":
        X = np.random.rand(n, feat_dim).astype(np.float32)
    elif mode == "degree_onehot":
        deg = np.array([G.degree(i) for i in G.nodes()])
        bins = np.clip(deg, 0, feat_dim - 1)
        X = np.zeros((n, feat_dim), dtype=np.float32)
        X[np.arange(n), bins] = 1.0
    else:
        raise ValueError("Unknown feature mode")
    return X

def generate_permuted_graph_pair(G, X):
    nodes = list(G.nodes())
    perm = np.random.permutation(nodes)

    Gs = G.copy()
    mapping = {nodes[i]: int(perm[i]) for i in range(len(nodes))}
    Gt = nx.relabel_nodes(G, mapping)

    src_nodes = sorted(Gs.nodes())
    tgt_nodes = sorted(Gt.nodes())

    gt_map = {i: mapping[i] for i in src_nodes}

    Xt = np.zeros_like(X)
    for i in src_nodes:
        Xt[gt_map[i]] = X[i]

    Xs = X.copy()
    return Gs, Gt, Xs, Xt, gt_map

def graph_to_adjacency(G):
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float32)
    return A, nodes

def build_gt_index_map(src_nodes, tgt_nodes, gt_map):
    tgt_index = {node: idx for idx, node in enumerate(tgt_nodes)}
    gt_idx = {i: tgt_index[gt_map[i]] for i in src_nodes}
    return gt_idx

def prepare_dataset(dataset_name="karate", feat_dim=16):
    G = load_base_graph(dataset_name)
    X = generate_node_features(G, feat_dim=feat_dim, mode="random")
    return G, X