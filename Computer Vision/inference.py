"""
Inference script for Vesuvius Surface Detection Model
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from vesuvius_model import UNet3D


def load_model(checkpoint_path, device='cpu'):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    model = UNet3D(in_ch=1, base_ch=32, out_ch=1).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using untrained model.")
    
    model.eval()
    return model


def predict_volume(model, volume, device='cpu', batch_size=1, threshold=0.5):
    """
    Predict on a 3D volume.
    
    Args:
        model: Trained model
        volume: Input volume as numpy array or torch tensor [C, D, H, W] or [D, H, W]
        device: Device to run inference on
        batch_size: Batch size for inference
        threshold: Threshold for binary predictions
        
    Returns:
        Predictions as numpy array with same spatial dimensions as input
    """
    model.eval()
    
    # Convert to tensor if needed
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume).float()
    
    # Add channel dimension if missing
    if volume.dim() == 3:
        volume = volume.unsqueeze(0)
    
    # Add batch dimension if missing
    if volume.dim() == 4:
        volume = volume.unsqueeze(0)
    
    # Move to device
    volume = volume.to(device)
    
    # Get original spatial dimensions
    original_shape = volume.shape[2:]
    
    # Make predictions
    with torch.no_grad():
        predictions = model(volume)
        
        # Apply threshold for binary predictions
        binary_predictions = (predictions > threshold).float()
        
        # Convert back to numpy
        if binary_predictions.dim() == 5:
            binary_predictions = binary_predictions.squeeze(0)
        if binary_predictions.dim() == 4:
            binary_predictions = binary_predictions.squeeze(0)
        
        predictions_np = binary_predictions.cpu().numpy()
        probabilities_np = predictions.squeeze().cpu().numpy() if predictions.dim() > 3 else predictions.cpu().numpy()
    
    return predictions_np, probabilities_np


def predict_patches(model, volume, patch_size=(64, 64, 64), overlap=0.5, device='cpu', threshold=0.5):
    """
    Predict on large volumes by processing patches with overlap.
    
    Args:
        model: Trained model
        volume: Input volume as numpy array [D, H, W] or [C, D, H, W]
        patch_size: Size of patches to extract (D, H, W)
        overlap: Overlap ratio between patches (0 to 1)
        device: Device to run inference on
        threshold: Threshold for binary predictions
        
    Returns:
        Full volume predictions
    """
    model.eval()
    
    # Convert to numpy if needed
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()
    
    # Add channel dimension if missing
    if volume.ndim == 3:
        volume = volume[np.newaxis, :]
    
    # Get volume dimensions
    C, D, H, W = volume.shape
    
    # Calculate step size
    step_d = int(patch_size[0] * (1 - overlap))
    step_h = int(patch_size[1] * (1 - overlap))
    step_w = int(patch_size[2] * (1 - overlap))
    
    # Initialize output arrays
    pred_volume = np.zeros((D, H, W), dtype=np.float32)
    count_volume = np.zeros((D, H, W), dtype=np.float32)
    
    # Process patches
    print("Processing patches...")
    for d in range(0, D - patch_size[0] + 1, step_d):
        for h in range(0, H - patch_size[1] + 1, step_h):
            for w in range(0, W - patch_size[2] + 1, step_w):
                # Extract patch
                patch = volume[:, d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]]
                
                # Predict
                pred_patch, _ = predict_volume(model, patch, device=device, threshold=threshold)
                
                # Accumulate predictions
                if pred_patch.ndim == 3:
                    pred_volume[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += pred_patch
                    count_volume[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += 1
                else:
                    pred_volume[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += pred_patch.squeeze()
                    count_volume[d:d+patch_size[0], h:h+patch_size[1], w:w+patch_size[2]] += 1
    
    # Average overlapping regions
    count_volume[count_volume == 0] = 1  # Avoid division by zero
    pred_volume = pred_volume / count_volume
    
    # Apply threshold
    binary_volume = (pred_volume > threshold).astype(np.float32)
    
    return binary_volume, pred_volume


def main():
    """Example inference usage."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = "checkpoints/best_model.pth"
    model = load_model(checkpoint_path, device=device)
    
    # Example: Load your volume data
    # volume = np.load("path/to/your/volume.npy")  # Shape: [D, H, W] or [C, D, H, W]
    
    # For demonstration, create dummy data
    print("\nGenerating dummy data for demonstration...")
    volume = np.random.randn(128, 128, 128).astype(np.float32)
    
    # Predict on full volume (for small volumes)
    print("\nRunning inference...")
    predictions, probabilities = predict_volume(model, volume, device=device, threshold=0.5)
    
    print(f"Input shape: {volume.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Prediction stats:")
    print(f"  Positive voxels: {np.sum(predictions > 0):,} ({100*np.mean(predictions > 0):.2f}%)")
    print(f"  Probability range: [{probabilities.min():.4f}, {probabilities.max():.4f}]")
    
    # For large volumes, use patch-based prediction
    # print("\nRunning patch-based inference...")
    # predictions_patches, probabilities_patches = predict_patches(
    #     model, volume, patch_size=(64, 64, 64), overlap=0.5, device=device, threshold=0.5
    # )
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()
