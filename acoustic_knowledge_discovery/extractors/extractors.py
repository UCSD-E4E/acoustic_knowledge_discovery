from ABC import abstract

import datasets

from ..dataset import KnowledgeDataset

class Extractor():
    """Extracts some data and formats into useful output

    Takes raw data (such as an ecological dataset like kaleidoscope or raven)
    Outputs a KnowledgeDataset
    See `../dataset.py`
    """

    def __init__(
            filename_col:str, ....) #Going to add stuff about files here

    def forward() -> KnowledgeDataset:
        raise NotImplemented()
