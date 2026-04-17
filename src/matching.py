import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

def similarity_matrix(Zs, Zt):
    return cosine_similarity(Zs, Zt)


def match_nearest_neighbor(sim):
    return sim.argmax(axis=1)


def match_hungarian(sim):
    row_ind, col_ind = linear_sum_assignment(-sim)
    pred = np.zeros(sim.shape[0], dtype=int)
    pred[row_ind] = col_ind
    return pred