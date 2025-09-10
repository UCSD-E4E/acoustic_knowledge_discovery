from ..dataset import ChunkDataset

import numpy as np
from scipy.linalg import toeplitz
from scipy.stats import zscore, entropy
from statsmodels.tsa.stattools import acf

def Entropy(p1: np.ndarray) -> np.ndarray:
    """
    From Colonna et. al., calculates von Neumann Entropy using SciPy's
    Shannon's Entropy implementation
    """
    p1 = p1/np.sum(p1)
    return entropy(p1)/np.log(len(p1))

def JSD(p: np.ndarray, q=None) -> tuple[float]:
    """
    Calculates Jensen-Shannon Divergence
    """
    
    n = len(p)
    if q is None:
        q = np.ones(n)/n # Uniform reference
    elif type(q) is not np.ndarray:
        raise "Bad type for equilibrium distributions"
    elif len(q) != n:
        raise "Distributions are not the same size"
    else:
        q = q/q.sum() # normalize q
    
    p = np.asarray(p)
    q = np.asarray(q)
    p = p/p.sum() # normalize
    m = (p + q) / 2
    
    jensen0 = -2*((((n+1)/n)*np.log(n+1)-2*np.log(2*n) + np.log(n))**(-1))
    
    return jensen0*(entropy(p, m) + entropy(q, m)) / 2

# https://github.com/juancolonna/EGCI/blob/master/Example_of_EGCI_calculation.ipynb
    
def process_batch(batch, lag):
    """
    Calculates EGCI in batch according to the method outlined in Colonna et. al. (2020)
    
    Parameters
    ----------
    x : `np.ndarray`
        An audio 
    lag : `int`
        t_max from the paper, the maximum value of t, the number of
        milleseconds shifted in autocorrelation

    Returns
    -------
    von Neumann Entropy, EGCI for the given audio
    """
    
    audio_batch = batch['raw_audio']
    
    egci_batch = []
    
    for raw_audio in audio_batch:
        x = zscore(raw_audio)
        
        # Algorithm steps 
        rxx = acf(x, nlags=lag, adjusted=True, fft=True)
        
        #https://github.com/blue-yonder/tsfresh/issues/902
        Sxx = toeplitz(rxx)
        s = np.linalg.svd(Sxx)[1] #svd(Sxx)
        
        entropy, complexity = Entropy(s), Entropy(s)*JSD(s)
        
        egci_batch.append([entropy, complexity])
    
    batch.update({"EGCI": egci_batch})
    
    return batch


class EGCI():
    """Gets EGCI for the knowledge graph
    
    Calculates the Environmental Global Complexity Index (EGCI)
    for a given audio clip based on methodology from Colonna et. al. 2020
    
    Parameters
    ----------
        nlag(int): _t_max_ in the original paper, determines the autocorrelation width
    
    """
    def __init__(self, nlag=512):
        self.nlag = nlag
        

    def __call__(self, chunkDS: ChunkDataset, num_proc=4) -> ChunkDataset:
        """
        Parameters
        ----------
            knowledge_ds (KnowledgeDataset): _description_

        Returns
        -------
            KnowledgeDataset: _the same dataset but with EGCI column_ `tuple(entropy, complexity)`
        """
        chunk_ds = chunkDS.chunk_ds
        
        #TODO ADD raw_audio COLUMN TO anno_ds containing audio from that split
        
        chunk_ds_processed = chunk_ds.map(process_batch, fn_kwargs={"lag": self.nlag}, batched=True, num_proc=num_proc)
        
        return ChunkDataset(chunk_ds=chunk_ds_processed)
    
        
        