# Tomato Disease Classification Project

This project implements multiple deep learning models to classify tomato leaf diseases using TensorFlow/Keras. The project compares the performance of a custom CNN architecture against transfer learning approaches using ResNet50V2 and EfficientNetB0.

## Setup Instructions

### 1. Create and Activate Virtual Environment

On Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

On macOS/Linux:
```bash
# Create virtual environment
python -m venv venv
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install tensorflow kaggle pandas numpy pillow matplotlib
```

### 3. Kaggle Authentication Setup

1. Create a Kaggle account if you don't have one
2. Go to your Kaggle account settings (https://www.kaggle.com/account)
3. Click on "Create New API Token"
4. Download the `kaggle.json` file
5. Create the `.kaggle` directory:
   - Windows: `mkdir %USERPROFILE%\.kaggle`
   - macOS/Linux: `mkdir ~/.kaggle`
6. Move the downloaded `kaggle.json` to the `.kaggle` directory
7. Set appropriate permissions (macOS/Linux only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 4. Project Structure

```
tomato-disease-classification/
├── venv/
├── models.py               # Contains model implementations
├── model_results.json      # Training results and metrics
├── model_comparison.png    # Performance comparison plots
├── README.md
└── trained_models/        # Directory containing saved models
```

## Running the Project

1. Make sure your virtual environment is activated
2. Run the main script:
```bash
python models.py
```

The script will:
- Download the dataset from Kaggle automatically
- Train three different models:
  - Basic CNN
  - ResNet50V2 (transfer learning)
  - EfficientNetB0 (transfer learning)
- Generate comparison plots
- Save the trained models
- Output performance metrics

## Output Files

The script generates several files:
- `model_comparison.png`: Visual comparison of model performance
- `model_results.json`: Detailed metrics for each model
- Saved model files:
  - `tomato_disease_basic_cnn.keras`
  - `tomato_disease_resnet50v2.keras`
  - `tomato_disease_efficientnetb0.keras`

## Dataset

The project uses the "Tomato Disease Multiple Sources" dataset from Kaggle:
https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources

## System Requirements

- Python 3.8 or higher
- At least 8GB RAM
- NVIDIA GPU recommended but not required
- Internet connection for downloading the dataset

## Troubleshooting

1. If you encounter memory issues:
   - Reduce batch_size in the code
   - Use a smaller input image size
   - Run on a machine with more RAM

2. If Kaggle authentication fails:
   - Verify your `kaggle.json` file is in the correct location
   - Check file permissions
   - Ensure you're using a valid API token

3. For GPU-related issues:
   - Verify TensorFlow can see your GPU
   - Update GPU drivers
   - Check CUDA and cuDNN compatibility with your TensorFlow version
