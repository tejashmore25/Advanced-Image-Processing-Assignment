import torch
import torch.nn as nn
import numpy as np
from .encoder import *
from .decoder import *

class SemanticSegmentationModel(nn.Module):
    def __init__(self, decoder_type='progressive', num_classes = 2):
        super().__init__()
        self.num_classes = num_classes
        #check wheter preTrained is False anywhere
        self.encoder = ResNet18Encoder()
        if decoder_type == 'single':
            self.decoder = SingleStageDecoder(num_classes=num_classes)
        elif decoder_type == 'progressive':
            self.decoder = ProgressiveDecoder(num_classes=num_classes)

    def forward(self, x):
        x2, x3, x4 = self.encoder(x)
        return self.decoder(x2, x3, x4)    