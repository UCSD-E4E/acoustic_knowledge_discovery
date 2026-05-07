from ..dataset import ChunkDataset
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize


def recommender_prune(
    edge_df,
    chunk_embeddings,
    keep_ratio=0.3,
    gamma=0.6
):
    """
    edge_df: pandas DataFrame with columns ['source', 'target', 'value']
             source = 'chunk_id_x'
             target = 'feature_value'
    chunk_embeddings: dict {chunk_id: embedding_vector}
    keep_ratio: percentage of edges to keep
    gamma: weight between collective and holistic importance
    """
    #extract chunk id from source
    edge_df = edge_df.copy()
    edge_df["chunk"] = edge_df["source"].str.replace("chunk_id_", "")

    #build feature to chunks mapping
    feature_groups = edge_df.groupby("target")["chunk"].apply(list)

    collective_scores = {}
    holistic_scores = {}

    for feature, chunks in feature_groups.items():

        #holistic importance (inverse degree)
        degree = len(chunks)
        holistic_scores[feature] = 1.0 / (degree + 1e-6)

        #collective importance (cohesion of chunk embeddings)
        vectors = [
            chunk_embeddings[c]
            for c in chunks
            if c in chunk_embeddings
        ]

        if len(vectors) < 2:
            collective_scores[feature] = 0.0
            continue

        vectors = np.array(vectors)
        vectors = normalize(vectors)

        #pairwise cosine similarity mean
        sim_matrix = np.dot(vectors, vectors.T)
        upper_tri = sim_matrix[np.triu_indices(len(vectors), k=1)]
        collective_scores[feature] = np.mean(upper_tri)

    #normalize scores to 0-1
    def normalize_dict(d):
        vals = np.array(list(d.values()))
        min_v, max_v = vals.min(), vals.max()
        return {
            k: (v - min_v) / (max_v - min_v + 1e-8)
            for k, v in d.items()
        }

    collective_scores = normalize_dict(collective_scores)
    holistic_scores = normalize_dict(holistic_scores)

    # Final feature score
    final_scores = {
        f: gamma * collective_scores[f] +
           (1 - gamma) * holistic_scores[f]
        for f in feature_groups.index
    }

    # Assign edge scores
    edge_df["score"] = edge_df["target"].map(final_scores)

    # Keep top edges
    n_keep = int(len(edge_df) * keep_ratio)
    edge_df = edge_df.sort_values("score", ascending=False)
    pruned_edges = edge_df.head(n_keep)

    return pruned_edges[["source", "target", "value"]]