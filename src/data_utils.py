import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)


def _largest_connected_component(G):
    """Keep only the largest connected component."""
    if G.number_of_nodes() == 0:
        raise ValueError("Graph is empty.")
    if nx.is_connected(G):
        return G
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def _read_edge_file(edge_path, src_col="src", dst_col="dst"):
    """
    Read an edge file into a NetworkX graph.
    Supported formats:
      - .csv  : must contain source/target columns
      - .edgelist / .txt : space/tab separated edgelist
    """
    edge_path = Path(edge_path)
    suffix = edge_path.suffix.lower()

    if not edge_path.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_path}")

    if suffix == ".csv":
        df = pd.read_csv(edge_path)

        # 如果没有 src/dst，就尝试自动识别前两列
        if src_col not in df.columns or dst_col not in df.columns:
            if len(df.columns) < 2:
                raise ValueError(
                    f"CSV edge file must contain at least 2 columns, got {df.columns.tolist()}"
                )
            src_col, dst_col = df.columns[:2]

        G = nx.from_pandas_edgelist(df, source=src_col, target=dst_col)

    elif suffix in [".edgelist", ".txt"]:
        # 默认按无权边表读取
        G = nx.read_edgelist(edge_path, nodetype=str)

    else:
        raise ValueError(f"Unsupported edge file format: {suffix}")

    return G


def load_base_graph(name="karate",
                    data_dir="../data/raw",
                    edge_file=None,
                    src_col="src",
                    dst_col="dst",
                    relabel=True,
                    keep_lcc=True):
    """
    Load a base graph.

    Parameters
    ----------
    name : str
        "karate" or "twitter"
    data_dir : str
        directory containing raw graph files
    edge_file : str or None
        optional edge filename for twitter
    src_col, dst_col : str
        source/target column names if CSV is used
    relabel : bool
        whether to relabel nodes into 0...n-1
    keep_lcc : bool
        whether to keep only the largest connected component

    Returns
    -------
    G : networkx.Graph
    """
    if name == "karate":
        G = nx.karate_club_graph()

    elif name == "twitter":
        data_dir = Path(data_dir)

        if edge_file is None:
            candidates = [
                data_dir / "twitter_edges.csv",
                data_dir / "twitter_edges.edgelist",
                data_dir / "twitter_edges.txt",
            ]
            edge_path = None
            for p in candidates:
                if p.exists():
                    edge_path = p
                    break
            if edge_path is None:
                raise FileNotFoundError(
                    f"Cannot find Twitter edge file in {data_dir}. "
                    f"Expected one of: twitter_edges.csv / twitter_edges.edgelist / twitter_edges.txt"
                )
        else:
            edge_path = data_dir / edge_file

        G = _read_edge_file(edge_path, src_col=src_col, dst_col=dst_col)

        # 转无向图、去重、自环清理
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        if keep_lcc:
            G = _largest_connected_component(G)

        if relabel:
            # 重编号为 0...n-1，方便后续 adjacency / feature 对齐
            G = nx.convert_node_labels_to_integers(G, ordering="sorted")

    else:
        raise NotImplementedError(f"Unknown dataset name: {name}")

    return G


def generate_node_features(G,
                           feat_dim=16,
                           mode="random",
                           attr_path=None,
                           node_id_col="node_id"):
    """
    Generate or load node features.

    Modes
    -----
    random:
        random Gaussian features
    degree_onehot:
        one-hot degree bins
    structural:
        graph structural features (degree, clustering, pagerank, core number, ...)
    from_file:
        load node features from CSV

    Returns
    -------
    X : np.ndarray, shape [n_nodes, d]
    """
    n = G.number_of_nodes()

    if mode == "random":
        X = np.random.randn(n, feat_dim).astype(np.float32)

    elif mode == "degree_onehot":
        deg = np.array([G.degree(i) for i in G.nodes()], dtype=np.int32)
        bins = np.clip(deg, 0, feat_dim - 1)
        X = np.zeros((n, feat_dim), dtype=np.float32)
        X[np.arange(n), bins] = 1.0

    elif mode == "structural":
        # 基础结构特征
        deg = np.array([G.degree(i) for i in G.nodes()], dtype=np.float32)
        clustering = np.array([nx.clustering(G, i) for i in G.nodes()], dtype=np.float32)
        pagerank_dict = nx.pagerank(G)
        pagerank = np.array([pagerank_dict[i] for i in G.nodes()], dtype=np.float32)
        core_dict = nx.core_number(G)
        core_num = np.array([core_dict[i] for i in G.nodes()], dtype=np.float32)

        base_feats = [deg, clustering, pagerank, core_num]
        X = np.stack(base_feats, axis=1)

        # 如果维度不够，补 degree one-hot
        if feat_dim > X.shape[1]:
            extra_dim = feat_dim - X.shape[1]
            bins = np.clip(deg.astype(int), 0, max(extra_dim - 1, 0))
            extra = np.zeros((n, extra_dim), dtype=np.float32)
            if extra_dim > 0:
                extra[np.arange(n), bins] = 1.0
                X = np.concatenate([X, extra], axis=1)

        elif feat_dim < X.shape[1]:
            X = X[:, :feat_dim]

        # 标准化
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
        X = X.astype(np.float32)

    elif mode == "from_file":
        if attr_path is None:
            raise ValueError("attr_path must be provided when mode='from_file'")

        attr_path = Path(attr_path)
        if not attr_path.exists():
            raise FileNotFoundError(f"Attribute file not found: {attr_path}")

        df = pd.read_csv(attr_path)

        if node_id_col not in df.columns:
            raise ValueError(
                f"Attribute file must contain a '{node_id_col}' column, got {df.columns.tolist()}"
            )

        # 假设图节点已经被 relabel 成 0...n-1
        df = df.sort_values(node_id_col).reset_index(drop=True)

        if len(df) != n:
            raise ValueError(
                f"Attribute rows ({len(df)}) do not match graph nodes ({n}). "
                f"Please ensure the attribute file matches the relabeled graph."
            )

        X = df.drop(columns=[node_id_col]).values.astype(np.float32)

    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    return X


def generate_permuted_graph_pair(G, X):
    """
    Generate an aligned graph pair by random node permutation.

    Returns
    -------
    Gs : source graph
    Gt : target graph (permuted)
    Xs : source features
    Xt : target features
    gt_map : dict, source node -> target node
    """
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
    """
    Convert graph to adjacency matrix using sorted node order.
    """
    nodes = sorted(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes, dtype=np.float32)
    return A, nodes


def build_gt_index_map(src_nodes, tgt_nodes, gt_map):
    """
    Build index-based ground-truth array for evaluation.

    Parameters
    ----------
    src_nodes : list
    tgt_nodes : list
    gt_map : dict
        source node -> target node

    Returns
    -------
    gt_idx : np.ndarray
        gt_idx[i] = j means source index i matches target index j
    """
    tgt_pos = {node: idx for idx, node in enumerate(tgt_nodes)}
    gt_idx = np.array([tgt_pos[gt_map[s]] for s in src_nodes], dtype=np.int64)
    return gt_idx