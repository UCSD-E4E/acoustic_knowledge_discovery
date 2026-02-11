from ..dataset import ChunkDataset
import librosa
import numpy as np
from pathlib import Path


def process_batch_temporal(batch, *, chunk_size):
    hop_length = 512
    frame_length = 1024

    batch_len = len(batch["file_path"])
    temporal_labels = []

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
            print(f"Error loading {file_path}, skipping: {exc}")
            temporal_labels.append("Unknown")
            continue

        if y.size == 0:
            temporal_labels.append("Silent")
            continue

        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)

        threshold = np.mean(rms) + np.std(rms)
        peaks = np.where(rms > threshold)[0]

        if len(peaks) < 2:
            temporal_labels.append("Steady")
            continue

        intervals = np.diff(peaks)

        if len(intervals) == 0:
            temporal_labels.append("Steady")
            continue

        cv = np.std(intervals) / (np.mean(intervals) + 1e-6)

        if cv < 0.4:
            temporal_labels.append("Rhythmic")
        else:
            temporal_labels.append("Bursty")

    return {"TemporalPattern": temporal_labels}


class TemporalPattern:
    def __init__(self, chunk_size: int):
        self.CHUNK_SIZE = chunk_size

    def __call__(self, chunkDS: ChunkDataset) -> ChunkDataset:
        chunk_ds = chunkDS.chunk_ds
        print("Computing Temporal Pattern feature...")

        processed_dataset = chunk_ds.map(
            process_batch_temporal,
            fn_kwargs={"chunk_size": self.CHUNK_SIZE},
            batched=True,
            batch_size=100,
            num_proc=4
        )

        return ChunkDataset(chunk_ds=processed_dataset)
