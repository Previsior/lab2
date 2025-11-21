import torch
import torch.nn as nn
import torch.nn.functional as F


class AEEncoder(nn.Module):
    """
    Lightweight convolutional encoder producing a latent vector.
    """

    def __init__(self, input_size: int = 32, latent_dim: int = 128):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        conv_out_size = input_size // 8
        self.flatten_dim = 128 * conv_out_size * conv_out_size
        self.fc = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        h = self.fc(x)
        return h

    @property
    def feature_dim(self) -> int:
        return self.latent_dim


class AEDecoder(nn.Module):
    """
    Upsampling decoder paired with AEEncoder.
    """

    def __init__(self, output_size: int = 32, latent_dim: int = 128):
        super().__init__()
        self.output_size = output_size
        conv_out_size = output_size // 8
        self.fc = nn.Linear(latent_dim, 128 * conv_out_size * conv_out_size)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        conv_out_size = self.output_size // 8
        x = self.fc(z)
        x = x.view(-1, 128, conv_out_size, conv_out_size)
        x = self.deconv(x)
        return x


class AutoEncoder(nn.Module):
    """Full autoencoder consisting of AEEncoder and AEDecoder."""

    def __init__(self, input_size: int = 32, latent_dim: int = 128):
        super().__init__()
        self.encoder = AEEncoder(input_size=input_size, latent_dim=latent_dim)
        self.decoder = AEDecoder(output_size=input_size, latent_dim=latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        recon = self.decoder(h)
        return recon, h

    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim
