import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CircleDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(
            0
        )  # Add channel dimension


def create_circle_image(image_size, radius, center_x, center_y):
    # Create a white image
    image = np.ones((image_size, image_size), dtype=np.uint8) * 255

    # Draw the circle (black color)
    center = (int(center_x), int(center_y))
    image = cv2.circle(image, center, int(radius), (0,), -1)

    return image.clip(0, 1)
