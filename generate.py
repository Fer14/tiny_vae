import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from vae import VAE, load_model

colors = [
    (235 / 255, 235 / 255, 211 / 255),  # Normalize RGB values to 0-1
    (8 / 255, 61 / 255, 119 / 255),  # Normalize RGB values to 0-1
]


custom_cmap = ListedColormap(colors, name="custom")


def visualize_single_latent(
    model,
    device,
    fixed_dims,
    fixed_values,
    varying_dim,
    latent_dim=3,
    grid_size=5,
    latent_range=(-1, 1),
    title="plot_latent_space.png",
):
    """
    Visualize a 5x1 plot by fixing two latent dimensions and varying the third.

    Args:
        model: Trained VAE model.
        device: Device (e.g., 'cpu' or 'cuda') to run the model.
        fixed_dims: List of two indices of the latent dimensions to fix (e.g., [0, 1]).
        fixed_values: List of two values to fix for the specified dimensions.
        varying_dim: Index of the latent dimension to vary (e.g., 2).
        latent_dim: Total number of latent dimensions (default: 3).
        grid_size: Number of steps to sample along the varying dimension.
        latent_range: Tuple specifying the range of the varying dimension (default: (-1, 1)).
    """
    model.eval()

    # Generate grid values for the varying dimension
    grid_values = np.linspace(latent_range[0], latent_range[1], grid_size)
    latent_vectors = []

    # Create latent vectors for the plot
    for value in grid_values:
        latent_vector = np.zeros(latent_dim)
        latent_vector[fixed_dims[0]] = fixed_values[0]  # Fix the first dimension
        latent_vector[fixed_dims[1]] = fixed_values[1]  # Fix the second dimension
        latent_vector[varying_dim] = value  # Vary the third dimension
        latent_vectors.append(latent_vector)

    latent_vectors = np.array(latent_vectors)

    # Decode the latent vectors into images
    generated_images = []
    with torch.no_grad():
        for z in latent_vectors:
            z_tensor = torch.tensor(z, dtype=torch.float32).to(device)
            generated_image = model.decoder(z_tensor)
            generated_images.append(generated_image.cpu().numpy())

    # Plot the 5x1 grid of images
    fig, axes = plt.subplots(1, grid_size, figsize=(15, 3))
    for i in range(grid_size):
        axes[i].imshow(
            generated_images[i].reshape(32, 32), cmap=custom_cmap
        )  # Assuming 32x32 images
        axes[i].axis("off")
        axes[i].set_title(f"{varying_dim} = {grid_values[i]:.2f}", fontsize=10)

    plt.suptitle(
        f"Latent Space Visualization (Fixed Dims {fixed_dims} = {fixed_values}, Varying Dim {varying_dim})",
        fontsize=16,
    )
    plt.tight_layout()
    # plt.show()
    plt.savefig(title)


def main():
    model = load_model(
        model=VAE(32 * 32, 3, 32 * 32), file_path="model.pth", device="cuda"
    )
    visualize_single_latent(
        model,
        "cuda",
        fixed_dims=[0, 1],
        fixed_values=[0, 0],
        varying_dim=2,
        grid_size=7,
        latent_range=(-3, 3),
        title="plots/plot1.png",
    )

    visualize_single_latent(
        model,
        "cuda",
        fixed_dims=[1, 2],
        fixed_values=[0, 0],
        varying_dim=0,
        grid_size=7,
        latent_range=(-3, 3),
        title="plots/plot2.png",
    )

    visualize_single_latent(
        model,
        "cuda",
        fixed_dims=[0, 2],
        fixed_values=[0, 0],
        varying_dim=1,
        grid_size=5,
        latent_range=(-3, 3),
        title="plots/plot3.png",
    )


if __name__ == "__main__":
    main()
