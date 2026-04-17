import numpy as np
import pandas as pd

from data_utils import generate_permuted_graph_pair, graph_to_adjacency, build_gt_index_map
from noise_utils import build_noisy_pair
from baselines import node2vec_embed, FINALAligner
from matching import similarity_matrix, match_nearest_neighbor
from metrics import accuracy_at_1, hit_at_k, timed_run


def run_experiments(G, X, edge_noise_levels, attr_noise_levels, runs=3, embed_dim=64):
    records = []

    for edge_noise in edge_noise_levels:
        for attr_noise in attr_noise_levels:
            for run in range(runs):
                Gs, Gt, Xs, Xt, gt_map = generate_permuted_graph_pair(G, X)
                Gs_n, Gt_n, Xs_n, Xt_n = build_noisy_pair(
                    Gs, Gt, Xs, Xt,
                    edge_noise=edge_noise,
                    attr_noise=attr_noise
                )

                As, src_nodes = graph_to_adjacency(Gs_n)
                At, tgt_nodes = graph_to_adjacency(Gt_n)
                gt_idx = build_gt_index_map(src_nodes, tgt_nodes, gt_map)

                # Node2Vec baseline
                (Zs, _), t1 = timed_run(node2vec_embed, Gs_n, embed_dim)
                (Zt, _), t2 = timed_run(node2vec_embed, Gt_n, embed_dim)
                sim = similarity_matrix(Zs, Zt)
                pred_nn = match_nearest_neighbor(sim)

                records.append({
                    "method": "Node2Vec+NN",
                    "edge_noise": edge_noise,
                    "attr_noise": attr_noise,
                    "run": run,
                    "acc": accuracy_at_1(pred_nn, gt_idx),
                    "hit1": hit_at_k(sim, gt_idx, 1),
                    "hit5": hit_at_k(sim, gt_idx, 5),
                    "runtime": t1 + t2
                })

                # FINAL baseline
                final = FINALAligner(alpha=0.8, max_iter=30).fit(As, At, Xs_n, Xt_n)
                S = final.predict_score_matrix()
                pred_final = final.predict_top1(use_hungarian=False)

                records.append({
                    "method": "FINAL-style",
                    "edge_noise": edge_noise,
                    "attr_noise": attr_noise,
                    "run": run,
                    "acc": accuracy_at_1(pred_final, gt_idx),
                    "hit1": hit_at_k(S, gt_idx, 1),
                    "hit5": hit_at_k(S, gt_idx, 5),
                    "runtime": np.nan
                })

    return pd.DataFrame(records)