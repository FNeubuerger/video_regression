"""
Video Regression Dataset module for temperature estimation.

This module provides a PyTorch Dataset class for loading image sequences
and extracting temperature labels from filenames for training CNN-LSTM models.
"""

import os
import re
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import List, Tuple, Optional
import random


class TemperatureSequenceDataset(Dataset):
    """
    PyTorch Dataset for loading image sequences with temperature labels.
    
    This dataset loads sequences of images from the data directory where each
    image filename contains the frame number and temperature information in the
    format: frame_{frame_number}_label_{temperature}.png
    
    Args:
        data_dir (str): Root directory containing sequence folders
        sequence_length (int): Number of consecutive frames to include in each sequence
        transform (callable, optional): Optional transform to be applied to images
        stride (int): Step size between consecutive sequences (default: 1)
        image_size (tuple): Target size for image resizing (height, width)
    """
    
    def __init__(self, data_dir="data", sequence_length=5, transform=None, 
                 stride=1, image_size=(128, 128)):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.stride = stride
        self.image_size = image_size
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Load and organize all image paths and their metadata
        self.sequences = self._load_sequences()
        
    def _load_sequences(self) -> List[Tuple[List[str], List[float]]]:
        """Load all valid sequences from the data directory."""
        sequences = []
        
        # Get all sequence directories
        sequence_dirs = [d for d in os.listdir(self.data_dir) 
                        if os.path.isdir(os.path.join(self.data_dir, d)) 
                        and d.startswith('sequence_')]
        
        for seq_dir in sequence_dirs:
            seq_path = os.path.join(self.data_dir, seq_dir)
            
            # Get all image files in this sequence
            image_files = [f for f in os.listdir(seq_path) 
                          if f.endswith('.png') and f.startswith('frame_')]
            
            # Parse filename to extract frame number and temperature
            frame_data = []
            for filename in image_files:
                frame_info = self._parse_filename(filename)
                if frame_info is not None:
                    frame_number, temperature = frame_info
                    frame_data.append((frame_number, temperature, 
                                     os.path.join(seq_path, filename)))
            
            # Sort by frame number
            frame_data.sort(key=lambda x: x[0])
            
            # Create sequences of consecutive frames
            for i in range(0, len(frame_data) - self.sequence_length + 1, self.stride):
                sequence_frames = frame_data[i:i + self.sequence_length]
                
                # Check if frames are consecutive or close enough
                frame_numbers = [f[0] for f in sequence_frames]
                if self._is_valid_sequence(frame_numbers):
                    image_paths = [f[2] for f in sequence_frames]
                    temperatures = [f[1] for f in sequence_frames]
                    sequences.append((image_paths, temperatures))
        
        print(f"Loaded {len(sequences)} sequences from {len(sequence_dirs)} sequence directories")
        return sequences
    
    def _parse_filename(self, filename: str) -> Optional[Tuple[int, float]]:
        """
        Parse filename to extract frame number and temperature.
        Expected format: frame_{frame_number}_label_{temperature}.png
        """
        pattern = r'frame_(\d+)_label_([0-9.]+)\.png'
        match = re.match(pattern, filename)
        
        if match:
            frame_number = int(match.group(1))
            temperature = float(match.group(2))
            return frame_number, temperature
        
        return None
    
    def _is_valid_sequence(self, frame_numbers: List[int]) -> bool:
        """Check if frame numbers form a valid sequence (allow some gaps)."""
        # Allow gaps of up to 5 frames between consecutive frames
        max_gap = 5
        for i in range(1, len(frame_numbers)):
            if frame_numbers[i] - frame_numbers[i-1] > max_gap:
                return False
        return True
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence of images and their corresponding temperatures.
        
        Returns:
            images: Tensor of shape (sequence_length, channels, height, width)
            temperatures: Tensor of shape (sequence_length,) containing temperature values
        """
        image_paths, temperatures = self.sequences[idx]
        
        # Load and transform images with optimizations
        images = []
        for img_path in image_paths:
            # Load image with memory optimization
            with Image.open(img_path) as image:
                # Convert grayscale to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Pre-resize for memory efficiency with large batches
                if hasattr(self, 'image_size') and image.size != self.image_size:
                    image = image.resize(self.image_size)
                
                # Apply transforms
                if self.transform:
                    image = self.transform(image)
                
                images.append(image)
        
        # Stack images into a tensor
        images_tensor = torch.stack(images)  # Shape: (sequence_length, channels, height, width)
        
        # Convert temperatures to tensor and take the mean for regression target
        temperatures_tensor = torch.tensor(temperatures, dtype=torch.float32)
        # Use mean temperature as the regression target
        target_temperature = temperatures_tensor.mean()
        
        return images_tensor, target_temperature
    
    def get_sample_info(self, idx: int) -> dict:
        """Get information about a specific sample for debugging."""
        image_paths, temperatures = self.sequences[idx]
        return {
            'index': idx,
            'image_paths': image_paths,
            'temperatures': temperatures,
            'sequence_length': len(image_paths)
        }


class TemperatureRegressionDataset(Dataset):
    """
    Simplified dataset that returns single temperature value per sequence.
    
    This version is more suitable for regression tasks where we want to predict
    a single temperature value from a sequence of images.
    """
    
    def __init__(self, data_dir="data", sequence_length=5, transform=None, 
                 stride=1, image_size=(128, 128), target_strategy='last'):
        """
        Args:
            target_strategy (str): How to determine target temperature
                - 'last': Use temperature of last frame in sequence
                - 'mean': Use mean temperature of all frames in sequence
                - 'first': Use temperature of first frame in sequence
        """
        self.base_dataset = TemperatureSequenceDataset(
            data_dir=data_dir, 
            sequence_length=sequence_length,
            transform=transform,
            stride=stride,
            image_size=image_size
        )
        self.target_strategy = target_strategy
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sequence of images and a single target temperature.
        
        Returns:
            images: Tensor of shape (sequence_length, channels, height, width)
            target_temp: Tensor containing single temperature value
        """
        images, temperatures = self.base_dataset[idx]
        
        # Determine target temperature based on strategy
        if self.target_strategy == 'last':
            target_temp = temperatures[-1]
        elif self.target_strategy == 'mean':
            target_temp = torch.mean(temperatures)
        elif self.target_strategy == 'first':
            target_temp = temperatures[0]
        else:
            raise ValueError(f"Unknown target_strategy: {self.target_strategy}")
        
        # Ensure target_temp is a scalar tensor
        target_temp = torch.tensor(target_temp, dtype=torch.float32)
        
        return images, target_temp
    
    def get_sample_info(self, idx: int) -> dict:
        """Get information about a specific sample for debugging."""
        base_info = self.base_dataset.get_sample_info(idx)
        images, target_temp = self[idx]
        
        base_info.update({
            'target_strategy': self.target_strategy,
            'target_temperature': target_temp.item(),
            'image_shape': images.shape
        })
        return base_info


def create_data_loaders(data_dir="data", batch_size=8, sequence_length=5, 
                       train_split=0.8, image_size=(128, 128), 
                       target_strategy='last', num_workers=2):
    """
    Create train and validation data loaders.
    
    Args:
        data_dir (str): Path to data directory
        batch_size (int): Batch size for data loaders
        sequence_length (int): Number of frames per sequence
        train_split (float): Fraction of data to use for training
        image_size (tuple): Target image size (height, width)
        target_strategy (str): Temperature target strategy
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    from torch.utils.data import DataLoader, random_split
    
    # Create dataset
    dataset = TemperatureRegressionDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        image_size=image_size,
        target_strategy=target_strategy
    )
    
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    dataset = TemperatureRegressionDataset(
        data_dir="data",
        sequence_length=5,
        image_size=(128, 128),
        target_strategy='last'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test a sample
        images, temperature = dataset[0]
        print(f"Sample 0:")
        print(f"  Images shape: {images.shape}")
        print(f"  Target temperature: {temperature.item():.2f}")
        
        # Show sample info
        info = dataset.get_sample_info(0)
        print(f"  Sample info: {info}")
        
        # Test data loader
        train_loader, val_loader = create_data_loaders(
            data_dir="data",
            batch_size=4,
            sequence_length=5
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
        
        # Test one batch
        for batch_images, batch_temps in train_loader:
            print(f"Batch images shape: {batch_images.shape}")
            print(f"Batch temperatures shape: {batch_temps.shape}")
            print(f"Temperature range: {batch_temps.min():.2f} - {batch_temps.max():.2f}")
            break
    else:
        print("No sequences found! Check your data directory structure.")