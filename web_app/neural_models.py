import torch
import torch.nn as nn


class StableKillPredictionNN(nn.Module):
    """Stable neural network architecture with bounded outputs"""
    
    def __init__(self, input_size: int):
        super(StableKillPredictionNN, self).__init__()
        
        self.layers = nn.Sequential(
            # Input layer
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(256),
            
            # Hidden layer 1
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            
            # Hidden layer 2
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            
            # Output layer with Sigmoid for bounded output
            nn.Linear(64, 1),
            nn.Sigmoid()  # Bounded 0-1, scale to 0-35
        )
        
        # Initialize weights conservatively
        self._init_weights()
    
    def _init_weights(self):
        """Conservative weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        output = self.layers(x)
        return output.squeeze() * 35.0  # Scale to 0-35 kills


class PrecisionTunedNN(nn.Module):
    """Precision-tuned neural network with residual connections"""
    
    def __init__(self, input_size: int):
        super(PrecisionTunedNN, self).__init__()
        
        # Main pathway
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 384),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.BatchNorm1d(384)
        )
        
        self.hidden1 = nn.Sequential(
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.BatchNorm1d(192)
        )
        
        self.hidden2 = nn.Sequential(
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.BatchNorm1d(96)
        )
        
        # Residual connection
        self.residual = nn.Linear(input_size, 96)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Linear(96, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Conservative weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Main pathway
        out = self.input_layer(x)
        out = self.hidden1(out)
        out = self.hidden2(out)
        
        # Residual connection
        residual = self.residual(x)
        out = out + residual
        
        # Output with scaling
        output = self.output(out)
        return output.squeeze() * 35.0  # Scale to 0-35 kills


class MassiveKillPredictionNN(nn.Module):
    """Large-scale neural network for maximum performance"""
    
    def __init__(self, input_size: int, hidden_sizes=[2048, 1024, 512, 256, 128, 64]):
        super(MassiveKillPredictionNN, self).__init__()
        
        layers = []
        current_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3 - i * 0.05),  # Decreasing dropout
                nn.BatchNorm1d(hidden_size)
            ])
            current_size = hidden_size
        
        # Output layer
        layers.extend([
            nn.Linear(current_size, 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        output = self.network(x)
        return output.squeeze() * 35.0  # Scale to 0-35 kills 