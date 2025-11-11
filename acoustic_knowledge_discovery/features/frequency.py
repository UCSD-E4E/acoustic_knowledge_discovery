from ..dataset import ChunkDataset
from typing import Dict, Tuple
import librosa
from pathlib import Path
import numpy as np

Bounds1D = Tuple[float, float]


def process_batch_efficiently(batch, *, chunk_size, bands: Dict[str, Bounds1D], threshold: float):
    n_fft = 2048
    hop_length = 512

    batch_len = len(batch["file_path"])
    frequency_hits = []

    for idx in range(batch_len):
        file_path = batch["file_path"][idx]
        start = batch["chunk_start"][idx]

        try:
            y, sr = librosa.load(
                Path(file_path),
                offset=float(start),
                duration=float(chunk_size),
                sr=None
            )
        except Exception as exc:
            print(f"Error loading audio file {file_path}, skipping: {exc}")
            frequency_hits.append([])
            continue

        if y.size == 0:
            frequency_hits.append([])
            continue

        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(D)
        power = magnitude**2
        S_db = librosa.power_to_db(power, ref=np.max)

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        avg_magnitude = np.mean(S_db, axis=1)
        mag_min = np.min(avg_magnitude)
        mag_max = np.max(avg_magnitude)
        denom = mag_max - mag_min
        if denom == 0:
            avg_magnitude_norm = np.zeros_like(avg_magnitude)
        else:
            avg_magnitude_norm = (avg_magnitude - mag_min) / denom

        if not (0 <= threshold <= 1):
            print(f"Warning: Threshold {threshold} is outside valid range [0, 1]. Using 0.5 instead.")
            threshold_to_use = 0.5
        else:
            threshold_to_use = threshold

        frequencies_for_row = []

        for band_name, (low_freq, high_freq) in bands.items():
            mask = (freqs >= low_freq) & (freqs < high_freq)
            if not np.any(mask):
                continue

            band_energy = float(np.mean(avg_magnitude_norm[mask])) #average normalized magnitude of the STFT within a band

            if band_energy > threshold_to_use:
                label = f"frequency_{band_name}"
                frequencies_for_row.append(label)

        frequency_hits.append(frequencies_for_row)

    return {"Frequency": frequency_hits}


class Frequency():
    def __init__(self, BIN_DICT: Dict[str, Bounds1D], THRESHOLD: float, chunk_size: int):
        self.BIN_DICT = BIN_DICT
        self.THRESHOLD = THRESHOLD
        self.CHUNK_SIZE= chunk_size
        

    def __call__(self, chunkDS: ChunkDataset) -> ChunkDataset:
        chunk_ds = chunkDS.chunk_ds
        print("Computing Frequency Activity feature...")
        
        processed_dataset = chunk_ds.map(
            process_batch_efficiently,
            fn_kwargs={"chunk_size": self.CHUNK_SIZE, "bands": self.BIN_DICT, "threshold": self.THRESHOLD},
            batched=True,
            batch_size=100, # change based on how much RAM a single batch needs
            num_proc=4      # Use parallel processes
        )
        
        return ChunkDataset(chunk_ds=processed_dataset)
