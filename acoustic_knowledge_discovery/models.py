import torch.nn as nn

import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torch

class AutoEncoder():
    
    def __init__(self, num_dims, model_state, device):
        super(AutoEncoder, self).__init__()
        
        self.device = device
    
        # architecture from https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0283396
        encoder = lambda nfeat : nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 1, 3, stride=2, padding=1),
            nn.Flatten(2),
            nn.Linear(64, nfeat)
        )

        decoder = lambda nfeat : nn.Sequential(
            nn.Linear(nfeat, 64),
            nn.Unflatten(2, (8,8)),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1, 256, (3, 3), bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), bias=False, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, (3, 3), bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, (3, 3), bias=False, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, (3, 3), bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, (3, 3), bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, (3, 3), bias=False, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, (3, 3), bias=False, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, (3, 3), bias=False, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Conv2d(1, 1, (3, 3), bias=False, padding=1)
        )
        
        self.encoder = encoder(num_dims)
        self.decoder = decoder(num_dims)

        self.model = nn.Sequential(
            self.encoder,
            self.decoder
        ).to(device)
        
        self.model.load_state_dict(model_state)
        self.model.eval().to(device)
        
    def embed(self, x):
        label = x.to(self.device)[..., :256]
        
        x = self.encoder(label)
        
        return x.detach().cpu().numpy().squeeze()
    
    def fit():
        #TODO: implement training?
        pass