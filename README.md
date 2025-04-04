# Neural Preset (Unofficial Implementation)

This is an unofficial implementation of the paper "Neural Preset for Color Style Transfer" (CVPR 2023) [[arXiv](https://arxiv.org/abs/2303.13511)].

We provide **~300 LUT files** for training!

## Overview

Neural Preset is a novel approach for color style transfer that learns to extract color styles from reference images and apply them to target images. This implementation provides a PyTorch Lightning-based training framework for the Neural Preset model.

## Setup

### Base Directory Setup

1. Set up the base directory structure:
```bash
# Create dataset directory in the parent folder
mkdir -p ../dataset/coco

# Make sure you are in the project root directory (neural-preset-public)
cd neural-preset
```

### Environment Setup

1. Create and configure the training server name in `configs/env.yaml`:
```bash
vi configs/env.yaml
```
Add the following line to the file:
```yaml
servername: 'your_training_server/computer_name'  # Change this to your training server/computer name
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

3. Download pretrained checkpoint:
```bash
# Create checkpoint directory
mkdir -p ckpts

# Download pretrained checkpoint from Google Drive
# Visit: [Pretrained Checkpoint](https://drive.google.com/open?id=1TZRVwIlzBBewwzgjrScrVzeynhBSLmm0&usp=drive_fs)
# Download the checkpoint file and place it in the ckpts directory
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

To start training, use the provided training script:
```bash
bash scripts/train_neural_preset.sh
```

The training script sets the following configurations:
- `CUDA_VISIBLE_DEVICES=1`: Use GPU 1 for training
- `model.name=neural_styler`: Use the Neural Styler model
- `model.ver=v1`: Use version 1 of the model
- `model.solver=v1`: Use version 1 of the solver
- `mode=train`: Set training mode

You can modify these parameters in the script or override them by adding command line arguments:
```bash
bash scripts/train_neural_preset.sh model.name=neural_styler model.ver=v2
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

## Testing with Pretrained Model

To test the model with the pretrained checkpoint:

1. Make sure you have downloaded the pretrained checkpoint and placed it in the `ckpts` directory.

2. Update the checkpoint path in the test script:
```bash
# Edit the test script
vi scripts/test_neural_preset.sh
```
Update the `load.ckpt_path` parameter to point to your downloaded checkpoint file.

3. Run the test script:
```bash
bash scripts/test_neural_preset.sh
```

The test script will:
- Load the pretrained model
- Process test images
- Generate color style transfer results
- Save the results in the results directory

## License

This implementation is for research purposes only. Please refer to the original paper's license for commercial use.
