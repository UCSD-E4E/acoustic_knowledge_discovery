import librosa
import numpy as np
import soundfile as sf
from scipy import signal

# Computes mel spectrogram from an audio signal
def compute_mel_spectrogram(y, sr, n_mels=128, hop_length=512):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB


# Converts a mel spectrogram in dB to an image format suitable for use in template matching or visualization
def spectrogram_to_image(S_dB):
    img = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    img = (img * 255).astype(np.uint8)
    return img


# Finds the dominant frequency range in a mel spectrogram based on the energy distribution
# Returns the mel bin indices that contain most of the energy
def find_dominant_frequency_range(spectrogram, energy_threshold=0.1):
    freq_energy = np.mean(spectrogram, axis=1)
    freq_energy = (freq_energy - freq_energy.min()) / (freq_energy.max() - freq_energy.min())
    
    dominant_bins = np.where(freq_energy > energy_threshold)[0]
    
    if len(dominant_bins) == 0:
        return 0, spectrogram.shape[0]

    return dominant_bins.min(), dominant_bins.max() + 1


# Extract a specific frequency range from the spectrogram to focus matching on relevant frequencies
def filter_spectrogram_by_frequency_range(spectrogram, freq_min, freq_max):
    return spectrogram[freq_min:freq_max, :]


# Efficiently load audio files with decimation to improve time
def fast_audio_load(audio_path, target_sr=22050):
    y, original_sr = sf.read(audio_path)
    
    if y.ndim > 1:
        y = y[:, 0]
    
    if original_sr != target_sr:
        decimation = original_sr // target_sr
        if decimation > 1 and len(y) > 100:
            try:
                y = signal.decimate(y, decimation)
                sr = original_sr // decimation
            except ValueError:
                y = librosa.resample(y, orig_sr=original_sr, target_sr=target_sr)
                sr = target_sr
        else:
            y = librosa.resample(y, orig_sr=original_sr, target_sr=target_sr)
            sr = target_sr
    else:
        sr = target_sr
    
    num_samples = len(y)
    duration_seconds = num_samples / sr
    return y, sr, duration_seconds


def insert_annotation_ds(ds, file_name, anno_start, anno_end, annotation, confidence, chunk_size):
    def add_hit(row):
        if (row["file_name"] == file_name 
            and (row["chunk_start"] + chunk_size) > anno_start 
            and row["chunk_start"] < anno_end):
            row["Annotation"] = row.get("Annotation", []) + [annotation]
            row["Confidence"] = row.get("Confidence", []) + [float(confidence)]
        return row
    
    return ds.map(add_hit)