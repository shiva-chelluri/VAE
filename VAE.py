# importing libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
from torch.utils.data import Dataset


# defining the variational-encoder
class VariationalEncoder(nn.Module):
    def __init__(self, input_size, latent_dims=50):
        super(VariationalEncoder, self).__init__()
        # defining encoding layers
        self.linear1 = nn.Linear(input_size[0], 192)
        self.linear2 = nn.Linear(192, 96)
        self.linear3 = nn.Linear(96, latent_dims)
        self.linear4 = nn.Linear(96, latent_dims)

        # defining re-parameterization
        self.Normal = torch.distributions.Normal(0, 1)
        self.Normal.loc = self.Normal.loc.cuda()
        self.Normal.scale = self.Normal.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        mu = F.relu(self.linear3(x))
        sigma = F.relu(self.linear4(x))

        z = mu + sigma * self.Normal.sample(mu.shape)
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


# defining the decoder

class Decoder(nn.Module):
    def __init__(self, input_size, latent_dims=50):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 96)
        self.linear2 = nn.Linear(96, 192)
        self.linear3 = nn.Linear(192, input_size[0])

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


# defining the autoencoder

class VAE(nn.Module):
    def __init__(self, input_size, latent_dims=50):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(input_size[0], latent_dims)
        self.decoder = Decoder(input_size[0], latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# defining the training function

def train(model, data, device, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for x, y in data:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat = model(x)
            loss = ((x_hat - y) ** 2).sum() + model.encoder.kl
            loss.backward()
            optimizer.step()
    return model

# defining the parquet to tensor setup


class Data(Dataset):
    def __init__(self, dataframe):
        self.data = torch.tensor(dataframe.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the same data as both input and output
        return self.data[idx], self.data[idx]
