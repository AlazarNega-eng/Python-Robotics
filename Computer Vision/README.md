# Vesuvius Surface Detection Model

A Computer Vision project implementing a 3D U-Net CNN model for surface/ink detection in volumetric CT scan data, based on the Vesuvius Ink Detection Challenge.

## Overview

This project implements a 3D U-Net architecture designed to process volumetric CT scan data and detect surface/ink regions. The model is inspired by winning solutions from the Vesuvius Ink Detection Challenge and is optimized for binary segmentation tasks on 3D medical imaging data.

## Architecture

The model uses a 3D U-Net architecture with:
- **Encoder**: 3 downsampling blocks with 3D convolutions, batch normalization, and ReLU activations
- **Bottleneck**: Double convolution block at the lowest resolution
- **Decoder**: 3 upsampling blocks with skip connections from the encoder
- **Output**: Single-channel binary segmentation mask with sigmoid activation

## Project Structure

```
Computer Vision/
├── vesuvius_model.py      # Model architecture definition
├── train_vesuvius.py      # Training script
├── inference.py           # Inference and prediction script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. For GPU support (recommended), install PyTorch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Training

1. Prepare your data in the required format (3D volumes with corresponding binary masks)
2. Update the `VesuviusDataset` class in `train_vesuvius.py` to load your data
3. Configure hyperparameters in the `main()` function
4. Run training:
```bash
python train_vesuvius.py
```

### Inference

1. Load a trained model checkpoint:
```python
from inference import load_model, predict_volume

model = load_model('checkpoints/best_model.pth', device='cuda')
```

2. Make predictions on your data:
```python
predictions, probabilities = predict_volume(model, volume, device='cuda', threshold=0.5)
```

For large volumes, use patch-based inference:
```python
from inference import predict_patches

predictions, probabilities = predict_patches(
    model, volume, patch_size=(64, 64, 64), overlap=0.5, device='cuda'
)
```

## Model Details

### Input Format
- Shape: `[Batch, Channels, Depth, Height, Width]`
- For single-channel CT scans: `[B, 1, D, H, W]`
- Expected data type: `float32`

### Output Format
- Shape: `[Batch, 1, Depth, Height, Width]`
- Values: Probabilities in range [0, 1] (sigmoid output)
- Binary predictions: Threshold at 0.5 (configurable)

### Loss Function
The model supports:
- Binary Cross-Entropy (BCE) Loss
- Combined BCE + Dice Loss (recommended for better performance)

### Training Hyperparameters
- Batch size: 2 (adjust based on GPU memory)
- Learning rate: 1e-4 (AdamW optimizer)
- Base channels: 32 (model size: ~2M parameters)

## References

This implementation is based on:
- Vesuvius Ink Detection Challenge winning solutions
- 3D U-Net architecture for volumetric segmentation
- Two-stage approaches using 3D CNNs + 2D segmentation networks

## Notes

- The dataset class (`VesuviusDataset`) is a template. You need to implement your actual data loading logic.
- For large volumes that don't fit in memory, use the patch-based inference method.
- Model checkpoints are saved in the `checkpoints/` directory.
- The model automatically uses CUDA if available, otherwise falls back to CPU.

## License

This project is provided for educational and research purposes.
