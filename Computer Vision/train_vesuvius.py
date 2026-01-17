"""
Training script for Vesuvius Surface Detection Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from vesuvius_model import UNet3D, BCEDiceLoss


class VesuviusDataset(Dataset):
    """
    Dataset class for Vesuvius CT scan data.
    
    This is a template dataset class. Replace with your actual data loading logic.
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # TODO: Load your actual data file list here
        # self.data_files = [...]
        
    def __len__(self):
        # TODO: Return actual dataset size
        return 100  # Placeholder
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Expected format:
        - data: 3D volume [C, D, H, W]
        - target: 3D binary mask [C, D, H, W]
        """
        # TODO: Load actual data here
        # Placeholder: Generate dummy data for demonstration
        data = torch.randn(1, 64, 64, 64)  # [C, D, H, W]
        target = torch.randint(0, 2, (1, 64, 64, 64)).float()
        
        if self.transform:
            data = self.transform(data)
            
        return data, target


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The neural network model
        loader: DataLoader for training data
        optimizer: Optimizer for model parameters
        criterion: Loss function
        device: Device to run on (cuda/cpu)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device)
        target = target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        
        # Calculate loss (model outputs sigmoid, so we need to handle it)
        # If using BCEDiceLoss with logits=False, remove sigmoid from model output
        # For this example, we'll use BCE on the output directly
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: The neural network model
        loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run on (cuda/cpu)
        
    Returns:
        Average loss for validation
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for data, target in pbar:
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def main():
    """Main training function."""
    # Hyperparameters
    batch_size = 2
    learning_rate = 1e-4
    num_epochs = 20
    save_dir = "checkpoints"
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize model
    model = UNet3D(in_ch=1, base_ch=32, out_ch=1).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss function and optimizer
    # Note: If using BCEDiceLoss, you may want to remove sigmoid from model output
    criterion = nn.BCELoss()  # Using BCELoss since model outputs sigmoid
    # Alternatively: criterion = BCEDiceLoss(weight_bce=0.5) (then remove sigmoid from model)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Data loaders
    # TODO: Replace with your actual data paths
    train_dataset = VesuviusDataset(data_dir="data/train", transform=None)
    val_dataset = VesuviusDataset(data_dir="data/val", transform=None)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50 + "\n")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (Val Loss: {val_loss:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
