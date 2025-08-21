from .feature_preprocessor import FeaturePreprocessor
import librosa
import numpy as np
import cv2 as cv
import os
import csv
import soundfile as sf
from scipy import signal
from datetime import datetime
from pydub import AudioSegment
from datasets import concatenate_datasets, Dataset
from ..dataset import KnowledgeDataset
from pathlib import Path


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
    
    return y, sr

# Get species name from xeno-canto filename format

def get_species_name(filename):
    base_name = filename.replace('.mp3', '').replace('.MP3', '').replace('.wav', '').replace('.WAV', '')
    parts = base_name.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return base_name



class TemplateMatching(FeaturePreprocessor):
    def __init__(self, CLIP_PATH: str | Path, TEMPLATE_PATH: str | Path, THRESHOLD: float):
        super().__init__("TemplateMatching")
        self.CLIP_PATH = CLIP_PATH
        self.TEMPLATE_PATH = TEMPLATE_PATH
        # self.anno_ds=kd.anno_ds
        # self.kd=kd
        self.THRESHOLD = THRESHOLD

    def __call__(self, kd: KnowledgeDataset) -> KnowledgeDataset:
        anno_ds = kd.anno_ds
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # # Initialize CSV file for storing all detection results
        # with open(self.CSV_OUTPUT_FILE, 'a', newline='') as csvfile: #ensures you start on next line 
        #     csv_writer = csv.writer(csvfile)
        #     #csv_writer.writerow(['template_name', 'clip_name', 'timestamp', 'score'])

        new_rows = {
            "file_path":[],
            "offset_time":[],
            "end_time": [],
            "annotation": [],
            "confidence": [],
            "ID": []
        }

        total_matches_count = 0

        # Use just one template per species to avoid redundancy
        template_files = [file for file in os.listdir(self.TEMPLATE_PATH) if file.lower().endswith(('.mp3', '.wav'))]
        species_templates = {}

        for template_file in template_files:
            species = get_species_name(template_file)
            if species not in species_templates:
                species_templates[species] = template_file

        selected_templates = list(species_templates.values())

        print(f"{len(template_files)} total templates")
        print(f"Filtered down to {len(selected_templates)} unique species templates")
        for species, template in species_templates.items():
            print(f"  {species}: {template}")

        # Iterate through each selected template
        template_count = 0
        for template_file in selected_templates:
                # Load and process the template audio with librosa
                template_name_clean = template_file.replace('.mp3', '').replace('.wav', '').replace('.MP3', '').replace('.WAV', '')
                template_path = os.path.join(self.TEMPLATE_PATH, template_file)
                try:
                    y_template, sr_template = librosa.load(template_path, sr=None)
                    template_spec = compute_mel_spectrogram(y_template, sr_template)
                    
                    # Find the frequency range where the template has the most energy
                    freq_min, freq_max = find_dominant_frequency_range(template_spec)
                    # Filter spectrogram to dominant frequencies and convert to image for template matching
                    template_spec_filtered = filter_spectrogram_by_frequency_range(template_spec, freq_min, freq_max)
                    template_img = spectrogram_to_image(template_spec_filtered)
                    
                except Exception as exception:
                    print(f"ERROR: Failed to load template {template_file}: {str(exception)}\n")
                    print(f"Skipping this template.\n\n")
                    continue
                
                
                # Track results for summary statistics
                clip_names = []
                match_counts = []
                
                # Process each audio clip in the clips directory
                for clip in os.listdir(self.CLIP_PATH):
                    if clip.lower().endswith('.wav'):
                        #f.write(f"Processing {clip}\n")
                        clip_path = os.path.join(self.CLIP_PATH, clip)
                        
                        try:
                            y_clip, sr_clip = fast_audio_load(clip_path, target_sr=22050)
                            #f.write(f"Loaded clip: {clip} (duration: {len(y_clip)/sr_clip:.1f}s)\n")
                        except Exception as exception:
                            print(f"ERROR: Failed to load {clip}: {str(exception)}\n")
                            print(f"Skipping this clip.\n\n")
                            continue
                        
                        # Process clip using same frequency filtering as template
                        clip_spec = compute_mel_spectrogram(y_clip, sr_clip)
                        #f.write(f"Clip shape (full): {clip_spec.shape}\n")
                        
                        clip_spec_filtered = filter_spectrogram_by_frequency_range(clip_spec, freq_min, freq_max)
                        clip_img = spectrogram_to_image(clip_spec_filtered)
                        #f.write(f"Clip shape (filtered): {clip_spec_filtered.shape}\n")

                        # Perform template matching using OpenCV
                        res = cv.matchTemplate(clip_img, template_img, cv.TM_CCOEFF_NORMED)
                        #f.write(f"Performed frequency-filtered template matching for {clip}\n")
                        
                        # Filter all matches above threshold
                        locations = np.where(res >= self.THRESHOLD)
                        matches = []
                        for pt in zip(*locations[::-1]):
                            score = res[pt[1], pt[0]]
                            matches.append((pt[1], pt[0], score))

                        matches.sort(key=lambda x: x[2], reverse=True)
                        #f.write(f"Found {len(matches)} matches above threshold {THRESHOLD}\n")

                        # Set the suppression_distance to half of the length of the template
                        template_length_frames = template_img.shape[1]
                        suppression_distance = int(template_length_frames)
                        #f.write(f"Suppression distance (frames): {suppression_distance}\n")

                        # Apply non-maximum suppression to avoid overlapping detections
                        selected = []
                        for y, x, score in matches:
                            if all(abs(x - xc) > suppression_distance for _, xc, _ in selected):
                                selected.append((y, x, score))

                        # Save all matches to CSV with timestamps
                        seconds_per_col = 512 / sr_clip
                        #with open(self.CSV_OUTPUT_FILE, 'a', newline='') as csvfile:
                        #csv_writer = csv.writer(csvfile)
                        for y, x, score in selected:
                            timestamp_match = x * seconds_per_col
                            audioTemplate = AudioSegment.from_file(template_path)
                            lenOfTemplate = len(audioTemplate) / 1000 

                            audioClip = AudioSegment.from_file(clip_path)
                            lenOfClip= len(audioClip) / 1000 
                            new_rows["file_path"].append(clip)
                            new_rows["offset_time"].append(float(timestamp_match))
                            new_rows["end_time"].append(float(min(timestamp_match + lenOfTemplate, lenOfClip)))
                            new_rows["annotation"].append(template_name_clean)
                            new_rows["confidence"].append(float(score))
                            new_rows["ID"].append(f"{clip}_{timestamp_match}")
                                
                            #csv_writer.writerow([clip,timestamp_match, min(timestamp_match + lenOfTemplate, lenOfClip), template_name_clean,  score])
                        
                        total_matches_count += len(selected)
                        
                        clip_names.append(clip)
                        match_counts.append(len(selected))

                # Update progress indicator
                template_count += 1
                progress_percent = (template_count / len(selected_templates)) * 100
                print(f"Progress: {template_count}/{len(selected_templates)} templates completed ({progress_percent:.1f}%)")

        # Print final summary to console
        if len(next(iter(new_rows.values()), [])) > 0:
            appended_anno_ds = Dataset.from_dict(new_rows)
            anno_ds_updated = concatenate_datasets([anno_ds, appended_anno_ds])
        else:
            anno_ds_updated = anno_ds  # nothing to append
        #print(f"Template matching completed. CSV matches saved to {self.CSV_OUTPUT_FILE}")
        print(f"Total matches found above the threshold of {self.THRESHOLD}: {total_matches_count}")
        #returns new knowledgedataset where anno_ds has template matching results appended
        return KnowledgeDataset(file_ds=kd.file_ds, anno_ds=anno_ds_updated)