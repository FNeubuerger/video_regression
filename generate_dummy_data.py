import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
from PIL import Image

class RandomImageDataset:
    """
    A class to generate and save random image sequences with regression labels in a folder structure
    compatible with torchvision.datasets.ImageFolder.

    Attributes:
        num_sequences (int): Number of sequences to generate.
        sequence_length (int): Number of frames per sequence.
        image_size (tuple): Size of the images (height, width).
        label_range (tuple): Range of regression labels (min, max).
    """
    def __init__(self, num_sequences, sequence_length, image_size, label_range, save_dir):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.label_range = label_range
        self.save_dir = save_dir

    def generate_and_save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        label_values = np.linspace(self.label_range[0], self.label_range[1], self.sequence_length)

        for seq_idx in tqdm(range(self.num_sequences), desc="Generating data"):
            sequence_dir = os.path.join(self.save_dir, f"sequence_{seq_idx}")
            os.makedirs(sequence_dir, exist_ok=True)

            for frame_idx, label in enumerate(label_values):
                # Generate random greyscale image
                # Start with a baseline black image
                image = np.zeros(self.image_size, dtype=np.uint8)
                
                # Add random noise to the image
                noise = np.random.randint(0, 50, self.image_size, dtype=np.uint8)
                image = np.clip(image + noise, 0, 255)
                # Add a bright spot corresponding to the temperature label
                center_x = int(self.image_size[1] / 2)
                center_y = int(self.image_size[0] / 2)
                std = 5
                x, y = np.meshgrid(np.arange(self.image_size[1]), np.arange(self.image_size[0]))
                gaussian = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * std**2))
                gaussian = (gaussian / gaussian.max()) * label  # Scale by the label value
                image = np.clip(image + gaussian, 0, 255).astype(np.uint8)
                
                image = Image.fromarray(image, mode="L")  # Convert to PIL Image
                image.save(os.path.join(sequence_dir, f"frame_{frame_idx}_label_{label:.2f}.png"))


# Example usage
if __name__ == "__main__":
    save_dir = "data"
    dataset_generator = RandomImageDataset(
        num_sequences=1000,
        sequence_length=10,
        image_size=(224, 224),
        label_range=(0.0, 100.0),
        save_dir=save_dir,
    )
    dataset_generator.generate_and_save()

    print(f"Dataset saved to {save_dir}")
