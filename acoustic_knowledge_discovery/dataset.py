from abc import ABC, abstractmethod

import datasets

class GraphDataset():
    """Formatted as
        file_path, offset, end_time, feat1, feat2, ....

        Output of feature pipeline
    """
    def __init__(ds, feature_cols=[]):
        raise NotImplementedError()



class KnowledgeDataset(datasets.DatasetDict):
    """ 
    HuggingFace DatasetDict with two splits
    (Input)
    - file_ds: ds of all file level features
        MUST CONTAIN
            - file_path
    - anno_ds: ds of all annotations
        MUST CONTAIN
            - file_path
            - offset_time
            - end_time
            - ID: `{file_path}_{offset}`

        has_property
            - label_cols
                each label col is 0 and 1 over the timestamp

    Args:
        DatasetDict (_type_): _description_
    """
    def __init__(self, file_ds, anno_ds):
        self.file_ds = file_ds
        self.anno_ds = anno_ds
        self.is_chunked = False

    def to_graph_format() -> GraphDataset:
        """Get the format for visualization

        1 dataset that contains rows formatted as 
        id, file_path, offset, end_time, feat1, feat2, ....

        This does not need to be overwritten
        Returns:
            datasets.dataset: _description_
        """

        #TODO WRITE THIS

