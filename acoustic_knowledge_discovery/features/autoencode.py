from ..dataset import ChunkDataset
import torch
from models import AutoEncoder

from pyha_analyzer.preprocessors import MelSpectrogramPreprocessor

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def process_batch(batch, model):
    """
    Calculates embeddings according to specified autoencoder
    """
    batch["auto_embedding"] = model.embed(batch["audio"])
    
    return batch


class autoencode():
    """Gets autoencoded embeddings from a specified model
    
    Parameters
    ----------
        model_state (str, optional): .pt file name for the model that is desired
        num_dims (int, optional): number of dimensions, must correspond with state_dict if provided,
        otherwise should be a preexisting trained model dimension [2, 10, 32]
        
    
    """
    def __init__(self, model, model_path=None, num_dims=None):
        model_list = {
            2: "dim_2.pt",
            10: "dim_10.pt",
            32: "dim_32.pt"
        }
        
        self.AutoEncoder()
        
        if model_path is None:
            if num_dims is None:
                raise ValueError("you must provide either a preexisting model or number of dims required")
            else:
                model_state = torch.load(model_list[num_dims])
        else:
            model_state = torch.load(model_path)
        
        
        
        self.model = AutoEncoder(num_dims, model_state, device)
        
        

    def __call__(self, chunkDS: ChunkDataset) -> ChunkDataset:
        """
        Parameters
        ----------
            knowledge_ds (KnowledgeDataset): _description_

        Returns
        -------
            KnowledgeDataset: _the same dataset but with EGCI column_ `tuple(entropy, complexity)`
        """
        chunk_ds = chunkDS.chunk_ds
        chunk_ds.set_transform(MelSpectrogramPreprocessor)
        
        #TODO ADD raw_audio COLUMN TO anno_ds containing audio from that split
        
        chunk_ds_processed = chunk_ds.map(process_batch, kw_args={"model": self.model}, batched=True, num_processes=4)
        
        return chunk_ds_processed