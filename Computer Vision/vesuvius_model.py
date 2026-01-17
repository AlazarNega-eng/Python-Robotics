"""
Vesuvius Surface Detection Model
A 3D U-Net CNN model for detecting ink/surface in volumetric CT scan data.
Based on the Vesuvius Ink Detection Challenge winning solutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """Double 3D Convolution Block with BatchNorm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super(DoubleConv3D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """
    3D U-Net Model for Volumetric Segmentation.
    
    Architecture:
    - Encoder: 3 downsampling blocks with DoubleConv3D
    - Bottleneck: DoubleConv3D block
    - Decoder: 3 upsampling blocks with skip connections
    
    Args:
        in_ch (int): Number of input channels (default: 1)
        base_ch (int): Base number of channels (default: 32)
        out_ch (int): Number of output channels (default: 1)
    """
    def __init__(self, in_ch=1, base_ch=32, out_ch=1):
        super(UNet3D, self).__init__()
        
        # Encoder
        self.enc1 = DoubleConv3D(in_ch, base_ch)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(base_ch, base_ch*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = DoubleConv3D(base_ch*2, base_ch*4)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv3D(base_ch*4, base_ch*8)
        
        # Decoder with skip connections
        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(base_ch*8, base_ch*4)
        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(base_ch*4, base_ch*2)
        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(base_ch*2, base_ch)
        
        # Output layer
        self.out_conv = nn.Conv3d(base_ch, out_ch, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass through the 3D U-Net.
        
        Args:
            x: Input tensor of shape [B, C, D, H, W]
            
        Returns:
            Output tensor of shape [B, out_ch, D, H, W] with sigmoid activation
        """
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder path with skip connections
        u3 = self.up3(b)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        # Output
        out = self.out_conv(d1)
        return torch.sigmoid(out)  # Output in [0,1] range
    
    def predict(self, x, threshold=0.5):
        """
        Make binary predictions on input data.
        
        Args:
            x: Input tensor
            threshold: Threshold for binary classification (default: 0.5)
            
        Returns:
            Binary predictions
        """
        with torch.no_grad():
            probs = self.forward(x)
            return (probs > threshold).float()


def dice_loss(pred, target, smooth=1.0):
    """
    Calculate Dice Loss for binary segmentation.
    
    Args:
        pred: Predicted probabilities
        target: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice loss value
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return 1 - (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice Loss for better segmentation performance.
    
    Args:
        weight_bce (float): Weight for BCE loss component (default: 0.5)
    """
    def __init__(self, weight_bce=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
    
    def forward(self, input, target):
        """
        Calculate combined BCE + Dice loss.
        
        Args:
            input: Model predictions (before sigmoid)
            target: Ground truth binary mask
            
        Returns:
            Combined loss value
        """
        bce_loss = self.bce(input, target)
        dice_l = dice_loss(torch.sigmoid(input), target)
        return self.weight_bce * bce_loss + (1 - self.weight_bce) * dice_l


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = UNet3D(in_ch=1, base_ch=32, out_ch=1).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    batch_size = 2
    depth, height, width = 64, 64, 64
    x = torch.randn(batch_size, 1, depth, height, width).to(device)
    
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
