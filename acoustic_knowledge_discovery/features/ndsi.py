from ..dataset import ChunkDataset
import numpy as np
import librosa
import os

def calculate_ndsi_score(magnitude_spec, freqs) -> float:
    """
    Calculation helper
    NDSI = (Biophony - Anthropophony) / (Biophony + Anthropophony)
    """
    # frequency bands
    anthro_min, anthro_max = 1000, 2000   # 1-2 kHz
    bio_min, bio_max = 2000, 11000        # 2-11 kHz

    # cmasks
    anthro_mask = (freqs >= anthro_min) & (freqs < anthro_max)
    bio_mask = (freqs >= bio_min) & (freqs < bio_max)

    # sum power; allow the mask to be empty (if SR is low), treating as 0
    anthro_power = np.sum(magnitude_spec[anthro_mask, :] ** 2) if np.any(anthro_mask) else 0.0
    bio_power = np.sum(magnitude_spec[bio_mask, :] ** 2) if np.any(bio_mask) else 0.0

    total_power = bio_power + anthro_power

    if total_power == 0:
        return 0.0

    return (bio_power - anthro_power) / total_power

def process_batch_ndsi(batch, chunk_size):
    """
    Loads audio, computes spectrogram, and calculates NDSI for batch.
    """
    ndsi_scores = []
    
    # batches of files and start times
    for path, start in zip(batch["file_path"], batch["chunk_start"]):
        try:
            # Load Audio
            y, sr = librosa.load(
                os.fspath(path),
                offset=float(start),
                duration=float(chunk_size),
                sr=None
            )

            # empty audio or load failure
            if y.size == 0:
                ndsi_scores.append(0.0)
                continue

            # Generate Spectrogram
            S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
            
            # Get Frequencies 
            freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

            # Calculate
            score = calculate_ndsi_score(S, freqs)
            ndsi_scores.append(score)

        except Exception as e:
            print(f"Error processing NDSI for {path} at {start}: {e}")
            ndsi_scores.append(0.0)

    # Return dictionary
    return {"NDSI": ndsi_scores}

class NDSI():
    """
    Calculates the Normalized Difference Soundscape Index (NDSI)
    """
    def __init__(self):
        pass

    def __call__(self, chunkDS: ChunkDataset, chunk_size: int, num_proc=4) -> ChunkDataset:
        chunk_ds = chunkDS.chunk_ds
        
        chunk_ds_processed = chunk_ds.map(
            process_batch_ndsi, 
            fn_kwargs={"chunk_size": chunk_size}, 
            batched=True, 
            num_proc=num_proc
        )
        
        return ChunkDataset(chunk_ds=chunk_ds_processed)