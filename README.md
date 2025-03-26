# Neural Preset (Unofficial Implementation)

This is an unofficial implementation of the paper "Neural Preset for Color Style Transfer" (CVPR 2023).

## Overview

Neural Preset is a novel approach for color style transfer that learns to extract color styles from reference images and apply them to target images. This implementation provides a PyTorch Lightning-based training framework for the Neural Preset model.

## Setup

### Base Directory Setup

1. Set up the base directory structure:
```bash
# Create dataset directory in the parent folder
mkdir -p ../dataset/coco

# Make sure you are in the project root directory (neural-preset-public)
cd neural-preset-public
```

### Environment Setup

1. Create and configure the training server name in `configs/env.yaml`:
```bash
vi configs/env.yaml
```
Add the following line to the file:
```yaml
servername: 'r6'  # Change this to your training server/computer name
```
This name will be used as a prefix for wandb experiment names.

### Dataset Setup

1. Download COCO dataset:
```bash
# Download and extract COCO dataset using the provided script
cd scripts
bash download_coco.sh

# Move the downloaded data to the correct location and reorganize
mv coco/images/train2017 ../../dataset/coco/train
mv coco/images/val2017 ../../dataset/coco/val
mv coco/images/test2017 ../../dataset/coco/test
mv coco/annotations ../../dataset/coco/
rm -r coco
```

The script will download:
- Training images (train2017) → train/
- Validation images (val2017) → val/
- Test images (test2017) → test/
- Annotations for all splits
- Stuff annotations

2. Download LUT files:
```bash
# Create LUT directory
mkdir -p datasets/luts

# Download LUT files from Google Drive
# Visit: https://drive.google.com/file/d/172j82XM9rwjIk-qYlN3VZAwk8sBy0XAf/view?usp=sharing
# Download cubes.zip and extract to datasets/luts/
unzip cubes.zip -d datasets/luts/
```

## Configuration

The project uses YAML configuration files for flexible parameter management:

### default.yaml
- Basic training settings
- Path configurations
- Logging settings
- Optimizer and scheduler configurations

### neural_styler.yaml
- Model architecture settings
- Training hyperparameters
- Dataset configurations
- Loss function parameters

Key configuration parameters:
```yaml
model:
  name: neural_styler
  ver: v1
  style_encoder: 'efficientnet-b0'  # Style encoder architecture
  k: 16                            # Number of style features

train:
  start_epoch: 0
  end_epoch: 32
  optimizer:
    mode: 'adam'
    adam:
      lr: 1e-4                     # Learning rate
  scheduler:
    mode: 'StepLR'                 # Learning rate scheduler

data:
  name: 'coco'
  root: '../../dataset/coco/images'
  batch_size: 32
  num_workers: 32
  size: 256                        # Input image size
  lut_root: '../datasets/luts'     # LUT files directory

criterion:
  lambda_consistency: 10           # Consistency loss weight
```

## Training

To start training:
```bash
python main.py
```

The training process will:
1. Load and preprocess the COCO dataset
2. Apply random LUT transformations for data augmentation
3. Train the Neural Preset model
4. Log metrics and visualizations to Weights & Biases
5. Save model checkpoints based on validation performance

## Visualization

Training progress can be monitored through:
- Weights & Biases dashboard
- Local visualization saves in the results directory
- Console output with loss values and progress bars

## License

This implementation is for research purposes only. Please refer to the original paper's license for commercial use.
