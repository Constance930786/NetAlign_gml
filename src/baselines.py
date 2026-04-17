import numpy as np
from node2vec import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
from matching import match_nearest_neighbor, match_hungarian

def node2vec_embed(G, dim=64, walk_length=10, num_walks=50, workers=1):
    n2v = Node2Vec(G, dimensions=dim, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = n2v.fit()
    nodes = sorted(G.nodes())
    Z = np.array([model.wv[str(n)] for n in nodes], dtype=np.float32)
    return Z, nodes

def normalize_adj(A):
    d = A.sum(axis=1)
    d_inv_sqrt = np.power(d + 1e-8, -0.5)
    D = np.diag(d_inv_sqrt)
    return D @ A @ D

class FINALAligner:
    def __init__(self, alpha=0.8, max_iter=30, tol=1e-6):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.S = None

    def fit(self, A1, A2, X1, X2):
        A1n = normalize_adj(A1)
        A2n = normalize_adj(A2)

        H = cosine_similarity(X1, X2).astype(np.float32)
        H = (H - H.min()) / (H.max() - H.min() + 1e-8)

        S = H.copy()
        for _ in range(self.max_iter):
            S_new = self.alpha * (A1n @ S @ A2n.T) + (1 - self.alpha) * H
            if np.linalg.norm(S_new - S) < self.tol:
                S = S_new
                break
            S = S_new

        self.S = S
        return self

    def predict_score_matrix(self):
        return self.S

    def predict_top1(self, use_hungarian=False):
        if use_hungarian:
            return match_hungarian(self.S)
        return match_nearest_neighbor(self.S)