import torch
import numpy as np

class SegmentationMetrics:
    def __init__(self, num_classes, device = 'cpu'):
        