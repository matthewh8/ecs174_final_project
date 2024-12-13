# Tomato Disease Classification Project

This project implements multiple deep learning models to classify tomato leaf diseases using TensorFlow/Keras. The project compares the performance of a custom CNN architecture against transfer learning approaches using ResNet50V2 and EfficientNetB0.

## Setup Instructions

### 1. Create and Activate Virtual Environment

On Windows:
```bash
# Create virtual environment
py -m venv venv
# Activate virtual environment
.\venv\Scripts\activate
```

On macOS/Linux:
```bash
# Create virtual environment
python3 -m venv venv
# Activate virtual environment
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install tensorflow pillow matplotlib numpy gdown pathlib scipy
```

### 3. Project Structure
```
tomato-disease-classification/
├── venv/
├── models/                 # Directory containing saved model files
├── plots/                 # Training visualizations and comparisons
├── results/              # JSON files with training metrics
├── corrupted_images/     # Any corrupted images found during verification
├── main.py              # Main script with model implementations
└── README.md
```

## Running the Project

1. Make sure your virtual environment is activated
2. Run the main script:
```bash
python main.py
```

The script will:
- Download the dataset from Google Drive automatically
- Verify image integrity and handle corrupted images
- Train three different models:
  - Basic CNN
  - ResNet50V2 (transfer learning)
  - EfficientNetB0 (transfer learning)
- Generate comparison plots
- Save the trained models
- Output performance metrics

## Output Files

The script generates several files with timestamps:

- `plots/model_comparison_[timestamp].png`: Visual comparison of model performance
- `results/model_results_[timestamp].json`: Detailed metrics for each model
- Saved model files in the `models/` directory:
  - `tomato_disease_basic_cnn_[timestamp].keras`
  - `tomato_disease_resnet50v2_[timestamp].keras`
  - `tomato_disease_efficientnetb0_[timestamp].keras`

## Dataset

The dataset will be automatically downloaded from Google Drive when running the script. The download process includes:
- Automatic download handling using `gdown`
- Image verification to detect and handle corrupted files
- Directory structure creation
- Dataset extraction

## Features

- Image integrity verification
- Automatic dataset download and extraction
- Multiple model architectures:
  - Custom CNN with batch normalization
  - Transfer learning with ResNet50V2
  - Transfer learning with EfficientNetB0
- Data augmentation for training
- Early stopping and model checkpointing
- Comprehensive performance comparison
- Automated results logging

## System Requirements

- Python 3.8 or higher
- At least 8GB RAM
- NVIDIA GPU recommended but not required
- Internet connection for downloading the dataset

## Model Architecture Details

### Basic CNN
- Three convolutional blocks with batch normalization
- Global average pooling
- Dense layers with dropout for regularization

### Transfer Learning Models
- Pre-trained ResNet50V2 and EfficientNetB0 base
- Custom top layers for classification
- Frozen pre-trained weights
- Dropout for regularization

## Troubleshooting

1. If you encounter memory issues:
   - Reduce `batch_size` (default is 32)
   - Reduce input shape (default is 128x128)
   - Free up system memory
   - Use a machine with more RAM

2. For download issues:
   - Check your internet connection
   - Verify the Google Drive file ID is still valid
   - Try downloading the dataset manually and place in the correct directory structure

3. For GPU-related issues:
   - Verify TensorFlow can access your GPU
   - Update GPU drivers
   - Check CUDA and cuDNN compatibility with your TensorFlow version

4. For corrupted images:
   - Check the `corrupted_images` directory for any files that were moved
   - Verify the integrity of the downloaded dataset
   - Try re-downloading the dataset

## Performance Monitoring

The training process includes:
- Real-time accuracy and loss monitoring
- Automatic early stopping to prevent overfitting
- Model checkpointing to save the best weights
- Comprehensive performance comparison plots
- Detailed JSON logs of training metrics

## Future Improvements

- Implement cross-validation
- Add model ensemble capabilities
- Expand to more disease classes
- Add prediction visualization tools
- Implement gradual unfreezing for transfer learning models
