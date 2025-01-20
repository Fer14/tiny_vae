import torch
import torch.nn as nn


def reparametrize(mu, log_var):
    variance = torch.exp(log_var)
    std = torch.sqrt(variance)

    epsilon = torch.randn_like(std)

    z = mu + epsilon * std
    return z


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),  # Output layer
            nn.Sigmoid(),  # Normalize output to [0, 1]
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(-1, 1, 32, 32)
        return x


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, output_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = reparametrize(mu, log_var)
        new_x = self.decoder(z)
        return new_x, mu, log_var


def vae_loss(new_x, x, mu, log_var):
    reconstruction_loss = nn.functional.binary_cross_entropy(
        new_x.view(-1, 32 * 32), x.view(-1, 32 * 32), reduction="sum"
    )

    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return reconstruction_loss + kl_loss


def save_model(model, file_path, device):
    # Move the model to CPU before saving, if it is on GPU
    model.to("cpu")
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")
    model.to(device)  # Move the model back to the original device


def load_model(model, file_path, device):
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {file_path}")
    return model
