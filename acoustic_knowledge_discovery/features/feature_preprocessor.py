from ..dataset import KnowledgeDataset

class FeaturePreprocessor():
    """Gets feature for the knowledge graph
    
    Perfered Naming Convention: (open to feed back)
        Augments anno_ds -> FeatureName_Chunk_FP
        Augments file_ds -> FeatureName_File_FP
    """
    def __init__(self):
        pass

    def __call__(self, knowledge_ds: KnowledgeDataset) -> KnowledgeDataset:
        """Adds to the KnowledgeDataset, either at a file level or a chunk level
        
        For The Young Devs
        fp = FeaturePreprocessor()
        fp() <- uses this call function

        
        MUST BE IMPLEMENTED
        Args:
            knowledge_ds (KnowledgeDataset): _description_

        Returns 
            KnowledgeDataset: The same dataset but with a new feature on top of it!
            The input should be a subset of the output
        """
        raise NotImplemented()