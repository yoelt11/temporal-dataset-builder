import torch
from torch import nn
from space_encoder import SpaceEncoder
from space_decoder import SpaceDecoder

class EncoderDecoder(nn.Module):
    def __init__(self, channel, latent_dim):
        super().__init__()
        self.encoder = SpaceEncoder(channel, latent_dim)
        self.decoder = SpaceDecoder(channel, latent_dim)

    def forward(self, X):
        Z = self.encoder(X)
        return self.decoder(Z)

if __name__ == "__main__":
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dummy input
    B, H, W, C, latent_dim = 300, 64, 256, 3, 16
    X = torch.randn(B,H,W,C).to(device)
    # initilize model
    model = EncoderDecoder(C, latent_dim).to(device)
    # run inference
    output = model(X)
    # Test output
    print("Network Output Shape: ", output.shape)
