# importing libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# --- 1. Model Definitions ---

class VariationalEncoder(nn.Module):
    """
    Encodes the input by outputting the parameters for a latent distribution.
    """

    def __init__(self, input_size=384, latent_dims=50):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_size, 192)
        self.linear2 = nn.Linear(192, 96)

        # Layers to output mu and log_var
        self.mu_layer = nn.Linear(96, latent_dims)
        self.log_var_layer = nn.Linear(96, latent_dims)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))

        # Get mu and log_var
        mu = self.mu_layer(x)
        log_var = self.log_var_layer(x)

        return mu, log_var


class Decoder(nn.Module):
    """
    Decodes the latent representation back to the original input space.
    """

    def __init__(self, input_size=384, latent_dims=50):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 96)
        self.linear2 = nn.Linear(96, 192)
        self.linear3 = nn.Linear(192, input_size)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        # No activation on the final layer for reconstruction
        reconstruction = self.linear3(z)
        return reconstruction


class VAE(nn.Module):
    """
    Combines the Encoder and Decoder into the full VAE model.
    """

    def __init__(self, input_size=384, latent_dims=50):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(input_size, latent_dims)
        self.decoder = Decoder(input_size, latent_dims)

    @staticmethod
    def reparameterize(mu, log_var):
        """
        Applies the parameterization trick to sample from the latent space.
        """
        std = torch.exp(0.5 * log_var)  # a.k.a. sigma
        eps = torch.randn_like(std)  # sample from standard normal
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


# --- 2. Data Handling ---

class ParquetDataset(Dataset):
    """
    Custom Dataset for loading data from a pandas DataFrame.
    """

    def __init__(self, dataframe):
        self.data = torch.tensor(dataframe.values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return the same data as both input and target
        return self.data[idx], self.data[idx]


# --- 3. Training Function ---

def train(model, data_loader, device, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()  # Set the model to training mode

    for epoch in range(epochs):
        total_loss = 0
        for x, y in data_loader:
            # Move data to the specified device (e.g., GPU)
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            x_hat, mu, log_var = model(x)

            # Calculate loss
            reconstruction_loss = F.mse_loss(x_hat, y, reduction='sum')
            kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconstruction_loss + kl_divergence

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}")

    return model
