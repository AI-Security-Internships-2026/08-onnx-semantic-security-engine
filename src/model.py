"""
Shared model definition for the ONNX Semantic Security Engine.

This module contains the ThreatMLP architecture used across
training, export, quantization, and inference.
"""

import torch.nn as nn


class ThreatMLP(nn.Module):
    """Multi-Layer Perceptron for network traffic threat classification.

    Architecture: Input → 128 → 64 → num_classes
    Activation: ReLU with 0.3 dropout between layers.
    """

    def __init__(self, input_dim, num_classes):
        super(ThreatMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x
