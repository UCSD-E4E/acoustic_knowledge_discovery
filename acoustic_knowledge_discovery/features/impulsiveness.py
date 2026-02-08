from ..dataset import ChunkDataset
from typing import Dict, Tuple, Optional
from pathlib import Path
import numpy as np
import librosa

Bounds1D = Tuple[float, float]

def process_batch_impulsiveness(
    batch,
    *,
    chunk_size: float,
    #bin_dict: Dict[str, Bounds1D],
):
    """
    Compute impulsiveness metrics for audio chunks based on the crest factor.
    Impulsiveness = ratio of peak amplitude to average amplitude (rms)
    """
    # get batch size and initialize lists for scores and bins
    batch_len = len(batch["file_path"])
    impulsiveness_scores = []

    #TODO: binning of impulsiveness scores. See below
    #impulsiveness_bins = []
    
    # Process each chunk in the batch
    for i in range(batch_len):
        file_path = batch["file_path"][i]

        # try to parse start time of chunk
        try:
            start_sec = int(batch["chunk_start"][i])
        except Exception:
            print(f"Warning: could not parse start time for row {i}, labled as -inf.")
            impulsiveness_scores.append(float("-inf"))
            #impulsiveness_bins.append(None)
            continue
        
        # try loading chunk
        try:
            y, sr = librosa.load(
                Path(file_path),
                offset=float(start_sec),
                duration=float(chunk_size),
                sr=None,
            )
        except Exception as exc:
            print(f"Error loading: {exc}")
            impulsiveness_scores.append(0.0)
            #impulsiveness_bins.append(None)
            continue
        
        # empty chunk handling
        if y.size == 0:
            impulsiveness_scores.append(0.0)
            #impulsiveness_bins.append(None)
            continue
        
        # Calculate impulsiveness metrics. 
        # Based on crest factor: peak amplitude / rms amplitude
        rms = np.sqrt(np.mean(y**2))
        peak = np.max(np.abs(y))
        
        # Crest Factor: peak-to-RMS ratio (higher = more impulsive)
        # Note: Crest factor is highest when theres 1 peak and low average amplitude
        # and lowest when the signal is more constant like white noise, or deep ses hum
        crest_factor = peak / (rms + 1e-10)
        
        
        impulsiveness_scores.append(float(crest_factor))
        
        #TODO: analyze dataset and create bins based on distribution of impulsiveness scores.
    
    return {
        "Impulsiveness_Score": impulsiveness_scores,
        #"Impulsiveness_Bin": impulsiveness_bins,
    }


# Adds impulsiveness score column to chunk dataset
class impulsiveness():
    def __init__(self, chunk_size: float):
        self.CHUNK_SIZE = chunk_size

    def __call__(self, chunkDS: ChunkDataset) -> ChunkDataset:
        chunk_ds = chunkDS.chunk_ds
        print("Adding impulsiveness score column to dataset...")

        # Apply to dataset with batching
        impulse_dataset = chunk_ds.map(
            process_batch_impulsiveness,
            fn_kwargs={
                "chunk_size": self.CHUNK_SIZE,
                #"bin_dict": {},  # TODO: add bin dict if we want to bin scores
            },
            batched=True,
            batch_size=100,  # process 100 examples at a time
            num_proc=4,      # use 4 parallel processes
        )

        return ChunkDataset(chunk_ds=impulse_dataset)