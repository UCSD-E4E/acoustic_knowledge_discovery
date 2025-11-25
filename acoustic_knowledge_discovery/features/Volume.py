from ..dataset import ChunkDataset
from typing import Dict, Tuple, Optional
import librosa
from pathlib import Path
import numpy as np

Bounds1D = Tuple[float, float]


def process_batch_volume_binned(
    batch,
    *,
    chunk_size: float,
    bin_dict: Dict[str, Bounds1D],
):
    """
    Compute per-chunk volume (in dB) and assign it to user-defined bins.

    bin_dict:
      Mapping from bin label -> (low_db, high_db), e.g.
      {
          "Low":    (-80.0, -40.0),
          "Medium": (-40.0, -25.0),
          "High":   (-25.0, 10.0),
      }

    Returns:
      {
        "Volume_dB":  [float, ...],
        "Volume_bin": [str | None, ...]
      }
    """
    batch_len = len(batch["file_path"])
    volumes_db = []
    volume_bins = []

    for idx in range(batch_len):
        file_path = batch["file_path"][idx]

        # Parse start time
        try:
            start_sec = int(batch["chunk_start"][idx])
        except Exception:
            print(f"Warning: could not parse start time for row {idx}, marking as silence.")
            volumes_db.append(float("-inf"))
            volume_bins.append(None)
            continue

        # Load the chunk
        try:
            y, _ = librosa.load(
                Path(file_path),
                offset=float(start_sec),
                duration=float(chunk_size),
                sr=None,  # keep original sampling rate
            )
        except Exception as exc:
            print(f"Error loading audio file {file_path}, skipping: {exc}")
            volumes_db.append(float("-inf"))
            volume_bins.append(None)
            continue

        # Empty chunk => silence
        if y.size == 0:
            volumes_db.append(float("-inf"))
            volume_bins.append(None)
            continue

        # RMS → dB
        rms = np.sqrt(np.mean(y**2))
        eps = 1e-10
        db = 20 * np.log10(rms + eps)
        db = float(db)
        volumes_db.append(db)

        # Assign bin (first matching bin in dict order)
        bin_label: Optional[str] = None
        for label, (low_db, high_db) in bin_dict.items():
            if low_db <= db < high_db:
                bin_label = label
                break

        volume_bins.append(bin_label)

    return {
        "Volume_dB": volumes_db,
        "Volume_bin": volume_bins,
    }

class Volume():
    def __init__(self, BIN_DICT: Dict[str, Bounds1D], chunk_size: float):
        """
        BIN_DICT: mapping from bin label -> (low_db, high_db).
                  Example:
                    {
                      "Low": (-80.0, -40.0),
                      "Medium": (-40.0, -25.0),
                      "High": (-25.0, 10.0),
                    }
        chunk_size: chunk duration in seconds.
        """
        self.BIN_DICT = BIN_DICT
        self.CHUNK_SIZE = chunk_size

    def __call__(self, chunkDS: ChunkDataset) -> ChunkDataset:
        chunk_ds = chunkDS.chunk_ds
        print("Computing Volume feature (binned)...")

        processed_dataset = chunk_ds.map(
            process_batch_volume_binned,
            fn_kwargs={
                "chunk_size": self.CHUNK_SIZE,
                "bin_dict": self.BIN_DICT,
            },
            batched=True,
            batch_size=100,  # adjust based on RAM
            num_proc=4,      # parallel processes
        )

        return ChunkDataset(chunk_ds=processed_dataset)



