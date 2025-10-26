from ..dataset import ChunkDataset
import torch
from ..models import AutoEncoder

import torchvision.transforms as transforms
from pyha_analyzer.preprocessors.preprocessors import PreProcessorBase

import numpy as np
import librosa
from torch import Tensor
import os


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



class MelSpectrogramPreprocessors(PreProcessorBase):
    def __init__(
        self,
        duration=5,
        augment=None,
        spectrogram_augments=None,
        class_list=[],
        n_fft=2048, 
        hop_length=256, 
        power=2.0, 
        n_mels=256,
        dataset_ref=None,
    ): 
        self.duration = duration
        self.augment = augment
        self.spectrogram_augments = spectrogram_augments

        # Below parameter defaults from https://arxiv.org/pdf/2403.10380 pg 25
        self.n_fft=n_fft
        self.hop_length=hop_length 
        self.power=power
        self.n_mels=n_mels

        super().__init__(name="MelSpectrogramPreprocessor")

    def __call__(self, batch):
        # print("process with melspec")
        new_audio = []
        new_labels = []
        pillow_transforms = transforms.ToPILImage()
        for item_idx in range(len(batch["file_path"])):
            path = batch["file_path"][item_idx]
            start = float(batch["chunk_start"][item_idx])
            y, sr = librosa.load(
                os.fspath(path),
                offset=start,
                duration=self.duration,
                sr=32_000,       
                mono=True
            )
            y = np.asarray(y, dtype=np.float32)
            sr = 32_000

            mels = np.array(
                pillow_transforms(
                    librosa.feature.melspectrogram(
                        y=y, sr=sr,
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        power=self.power,
                        n_mels=self.n_mels,
                    )
                ),
                dtype=np.float32
            )[np.newaxis, ...] / 255.0
            new_audio.append(mels)

        input_data = Tensor(np.concatenate(new_audio, axis=0)).unsqueeze(1)
        return input_data


class AutoEncoderProcessor():
    """Gets autoencoded embeddings from a specified model
    
    Parameters
    ----------
        model_state (str, optional): .pt file name for the model that is desired
        num_dims (int, optional): number of dimensions, must correspond with state_dict if provided,
        otherwise should be a preexisting trained model dimension [2, 10, 32]
        
    
    """
    def __init__(self, model_path=None, num_dims=None):
        
        model_list = {
            2: "dim_2.pt",
            10: "dim_10.pt",
            32: "dim_32.pt"
        }
        # print(device, model_path)
        # if model_path is None:
        #     if num_dims is None:
        #         raise ValueError("you must provide either a preexisting model or number of dims required")
        #     else:
        #         model_state = torch.load(model_list[num_dims], map_location=torch.device('cpu'))
        # else:
        model_state = torch.load(model_path, map_location=torch.device('cpu'))
        self.model_type = model_path
        
        
        self.model = AutoEncoder(num_dims, model_state, device)


    def process_batch(self, batch):
        """
        Calculates embeddings according to specified autoencoder
        """

        out = self.model.embed(batch)




        return {
            self.model_type + "_embeddings": out
        }    
        

    def __call__(self, chunkDS: ChunkDataset, chunk_size) -> ChunkDataset:
        """
        Parameters
        ----------
            knowledge_ds (KnowledgeDataset): _description_

        Returns
        -------
            KnowledgeDataset: _the same dataset but with EGCI column_ `tuple(entropy, complexity)`
        """
        test = MelSpectrogramPreprocessors(duration=chunk_size)
        chunkDS.chunk_ds.set_transform(test)
        
        #TODO ADD raw_audio COLUMN TO anno_ds containing audio from that split
        
        chunk_ds_processed = chunkDS.chunk_ds.map(self.process_batch, batched=True) #TODO fix this, num_processes=4

        col_name = chunk_ds_processed["train"].column_names[0]

        chunkDS.chunk_ds.reset_format()
        chunk_ds_processed.reset_format()

        chunkDS.chunk_ds["train"] = chunkDS.chunk_ds["train"].add_column(
            col_name, chunk_ds_processed["train"][col_name]
        )

        return chunkDS

       



        