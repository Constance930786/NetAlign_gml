import numpy as np
import time

def accuracy_at_1(pred, gt_idx):
    correct = 0
    for i, j in enumerate(pred):
        if gt_idx[i] == j:
            correct += 1
    return correct / len(pred)

def hit_at_k(score_matrix, gt_idx, k=1):
    topk = np.argsort(-score_matrix, axis=1)[:, :k]
    correct = 0
    for i in range(score_matrix.shape[0]):
        if gt_idx[i] in topk[i]:
            correct += 1
    return correct / score_matrix.shape[0]

def timed_run(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start