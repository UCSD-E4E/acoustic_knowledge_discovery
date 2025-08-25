from ..dataset import ChunkDataset
from abc import ABC, abstractmethod
import torch.nn as nn

class FeaturePreprocessor(nn.Module):
    """Base class for feature preprocessors in the knowledge discovery pipeline."""
    
    """Gets feature for the knowledge graph
    
    Perfered Naming Convention: (open to feed back)
        Augments anno_ds -> FeatureName_Chunk_FP
        Augments file_ds -> FeatureName_File_FP
    """
    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def forward(self, kd: ChunkDataset) -> ChunkDataset:
        """Transform and return a KnowledgeDataset."""
        ...
        
    @abstractmethod
    def __call__(self, knowledge_ds: ChunkDataset) -> ChunkDataset:
        """Adds to the KnowledgeDataset, either at a file level or a chunk level
        
        For The Young Devs
        fp = FeaturePreprocessor()
        fp() <- uses this call function

        
        MUST BE IMPLEMENTED
        Args:
            knowledge_ds (KnowledgeDataset): _description_

        Returns 
            KnowledgeDataset: The same dataset but with a new feature on top of it!
            The input SHOULD be a subset of the output

            Chunking is the ONLY exception
        """
        return self.forward(knowledge_ds)