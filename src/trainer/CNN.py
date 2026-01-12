import torch.nn as nn
import torch

class CustomCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, image_size: tuple) -> None:
        super(CustomCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
    
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.flatten_dim = 128

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        self.latent = x
        x = self.fc_layers(x)
        return x
    

    def get_latent_features(self, x):
        with torch.no_grad():
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)
        return x
