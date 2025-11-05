from ..dataset import KnowledgeDataset

import numpy as np
import librosa

def process_batch(batch, model):
    """
    Returns frequency bands from this document 
    (https://docs.google.com/document/d/1I49OEw9DfDSyOiA5v7q0A9dZIIwR1ALcGuyE4zhR7pc) 
    sorted by activity level
    """
    bands = []
    
    for audio, sr in zip(batch['audio'], batch['sample_rate']):
    
        freq_sum = np.sum(audio, axis=1)
        
        mel_freqs = librosa.mel_frequencies(256, fmax=sr/2)
        
        bands_hz = [[0, 2_500], [2_500, 4_000], [4_000, 8_000], [8000, 20_000], [20_000, 250_000]]
        band_labels = np.array(["low", "mid", "active", "high", "ultra"])
        
        label = band_labels[np.argsort([np.sum(freq_sum[((mel_freqs > band_min) == (mel_freqs < band_max))]) for band_min, band_max in bands_hz])[::-1]]
        
        bands.append(label)
    
    batch.update({"band": bands})
    
    return batch


class banding():
    """
    Gets high activity bands for a melspectrogram 
    
    """
    def __init__(self):
        pass
        

    def __call__(self, knowledge_ds: KnowledgeDataset) -> KnowledgeDataset:
        """
        Parameters
        ----------
            knowledge_ds (KnowledgeDataset): _description_

        Returns
        -------
            KnowledgeDataset: _the same dataset but with EGCI column_ `tuple(entropy, complexity)`
        """
        anno_ds = knowledge_ds.anno_ds
        anno_ds.set_transform(MelSpectrogramPreprocessor)
        
        #TODO ADD raw_audio COLUMN TO anno_ds containing audio from that split
        
        anno_ds_processed = anno_ds.map(process_batch, batched=True, num_processes=4)
        
        
        return anno_ds_processed