import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CircleDataset, create_circle_image
from vae import VAE, save_model, vae_loss


def train_vae(model, data_loader, optimizer, device, epochs=100):
    model.train()
    model.to(device)
    total_batches = len(data_loader) * epochs

    with tqdm(total=total_batches, desc="Training") as pbar:
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, data in enumerate(data_loader):
                data = data.to(device)
                optimizer.zero_grad()

                # Forward pass
                reconstructed_x, mu, log_var = model(data)

                # Compute the loss
                loss = vae_loss(reconstructed_x, data, mu, log_var)
                running_loss += loss.item()

                # Backward pass
                loss.backward()
                optimizer.step()

                # Update the progress bar with the latest loss
                pbar.set_description_str(f"Training Epoch {epoch + 1}/{epochs}")
                pbar.set_postfix(loss=running_loss / (batch_idx + 1))
                pbar.update(1)

    return model


def main():
    # Example usage
    image_size = 32  # Assuming 32x32 images
    latent_dim = 3  # Representing radius, x, y
    input_dim = image_size * image_size  # Flattened image
    output_dim = input_dim  # Reconstructed image size
    n_images = 500000

    # Generate some example images (use your dataset generation code here)
    images = []

    for i in range(n_images):
        radius = np.random.uniform(3, 10)  # Radius between 3 and 10 pixels
        center_x = np.random.uniform(radius, image_size - radius)  # X-coordinate
        center_y = np.random.uniform(radius, image_size - radius)  # Y-coordinate
        image = create_circle_image(image_size, radius, center_x, center_y)
        images.append(image)

    dataset = CircleDataset(images)
    data_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    vae = VAE(input_dim, latent_dim, output_dim)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the VAE
    model = train_vae(vae, data_loader, optimizer, device, epochs=500)
    save_model(model, "model.pth", device)


if __name__ == "__main__":
    main()
