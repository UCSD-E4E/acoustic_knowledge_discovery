import datasets
from datasets import DatasetDict

class GraphDataset():
    """Formatted as
        file_path, offset, end_time, feat1, feat2, ....

        Output of feature pipeline
    """
    def __init__(ds, feature_cols=[]):
        raise NotImplementedError()



class ChunkDataset(datasets.DatasetDict):
    def __init__(self, chunk_ds):
        self.chunk_ds = chunk_ds
    
    def to_graph_format() -> GraphDataset:
        """Get the format for visualization

        1 dataset that contains rows formatted as 
        id, file_path, offset, end_time, feat1, feat2, ....

        This does not need to be overwritten
        Returns:
            datasets.dataset: _description_
        """

        #TODO WRITE THIS
