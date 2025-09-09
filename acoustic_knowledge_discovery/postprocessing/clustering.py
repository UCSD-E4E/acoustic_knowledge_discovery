import numpy as np
from sklearn.cluster import KMeans



from ..dataset import ChunkDataset

class Cluster_API():
    def __init__(self, chunkDS, cluster_alg=KMeans, num_clusters=2):
        self.chunkDS = chunkDS
        self.cluster_alg = cluster_alg
        self.num_clusters = num_clusters

    def __call__(self, column_name):
        X = np.asarray(self.chunkDS.chunk_ds["train"][column_name], dtype=float)

        clusters = self.cluster_alg(
            n_clusters=self.num_clusters,
            random_state=0,
            n_init="auto").fit_predict(X)
        
        self.chunkDS.chunk_ds["train"] = self.chunkDS.chunk_ds["train"].add_column(
            f"{column_name}_clusters",
            clusters
        )
        return self.chunkDS