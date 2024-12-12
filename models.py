import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Input, 
    Dropout, BatchNormalization
)
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import gdown
import zipfile
from pathlib import Path
from datetime import datetime
from PIL import Image
import shutil

def verify_images(directory, move_corrupted=True):
    """
    Verify images and optionally move corrupted ones to a separate directory.
    Returns a list of corrupted file paths.
    """
    corrupted_dir = os.path.join(os.path.dirname(directory), 'corrupted_images')
    if move_corrupted and not os.path.exists(corrupted_dir):
        os.makedirs(corrupted_dir)

    corrupted_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepath = os.path.join(root, filename)
                try:
                    # Open and close the image explicitly
                    with open(filepath, 'rb') as f:
                        img = Image.open(f)
                        try:
                            img.verify()
                            # Need to reopen after verify
                            img = Image.open(f)
                            img.load()
                            img.close()
                        except Exception as e:
                            print(f"Corrupted image found: {filepath}")
                            print(f"Error: {str(e)}")
                            corrupted_files.append(filepath)
                            if move_corrupted:
                                dest = os.path.join(corrupted_dir, os.path.basename(filepath))
                                shutil.move(filepath, dest)
                                print(f"Moved to: {dest}")
                except Exception as e:
                    print(f"Error opening {filepath}: {str(e)}")
                    corrupted_files.append(filepath)
                    if move_corrupted:
                        dest = os.path.join(corrupted_dir, os.path.basename(filepath))
                        try:
                            shutil.move(filepath, dest)
                            print(f"Moved to: {dest}")
                        except Exception as e:
                            print(f"Error moving file: {str(e)}")
    
    return corrupted_files

def download_dataset():
    """
    Downloads the tomato disease dataset from Google Drive and extracts it.
    Includes error handling and verification of download.
    """
    FILE_ID = "1b2kdETIey4tSVlhn3ZJVGnKbZxSc1Sn_"
    OUTPUT_PATH = Path("dataset.zip")
    
    if not Path('train').exists():
        print("Downloading dataset...")
        try:
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            
            if not gdown.download(url, str(OUTPUT_PATH), quiet=False):
                raise Exception("Download failed")
            
            print("Extracting dataset...")
            with zipfile.ZipFile(OUTPUT_PATH, 'r') as zip_ref:
                zip_ref.extractall('.')
                
            OUTPUT_PATH.unlink()
            
            if not Path('train').exists() or not Path('valid').exists():
                raise Exception("Dataset extraction failed - missing train or valid directories")
                
            print("Dataset downloaded and extracted successfully")
            
        except Exception as e:
            print(f"Error downloading or extracting dataset: {str(e)}")
            if OUTPUT_PATH.exists():
                OUTPUT_PATH.unlink()
            raise
    else:
        print("Dataset already exists")

def create_basic_cnn(input_shape=(128, 128, 3), num_classes=4):
    """
    Creates a basic CNN model for image classification.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_resnet_model(input_shape=(128, 128, 3), num_classes=4):
    """
    Creates a ResNet50V2-based model using transfer learning.
    """
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

def create_efficientnet_model(input_shape=(128, 128, 3), num_classes=4):
    """
    Creates an EfficientNetB0-based model using transfer learning.
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

def plot_comparison(histories, model_names, save_path='plots'):
    """
    Plots and saves training history comparison between models.
    """
    Path(save_path).mkdir(exist_ok=True)
    metrics = ['accuracy', 'loss']
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for metric, ax in zip(metrics, axes):
        for model_name, history in zip(model_names, histories):
            ax.plot(history.history[metric], label=f'{model_name} (train)')
            ax.plot(history.history[f'val_{metric}'], label=f'{model_name} (val)')
        
        ax.set_title(f'Model {metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{save_path}/model_comparison_{timestamp}.png')
    plt.close()

def setup_data_generators(input_shape=(128, 128), batch_size=32):
    """
    Sets up data generators for training and validation with improved error handling.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    try:
        train_generator = train_datagen.flow_from_directory(
            'train',
            target_size=input_shape,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            'valid',
            target_size=input_shape,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    except Exception as e:
        print(f"Error setting up data generators: {str(e)}")
        raise

def main():
    try:
        # Create directories for outputs
        for directory in ['models', 'plots', 'results']:
            Path(directory).mkdir(exist_ok=True)
        
        # Download dataset first
        download_dataset()
        
        # Verify images after download/existence check
        print("Verifying dataset integrity...")
        corrupted_train = verify_images('train')
        corrupted_val = verify_images('valid')
        
        if corrupted_train or corrupted_val:
            print(f"Found and handled {len(corrupted_train)} corrupted training images")
            print(f"Found and handled {len(corrupted_val)} corrupted validation images")
        
        # Setup parameters
        batch_size = 32
        input_shape = (128, 128)
        
        # Prepare data generators
        train_generator, val_generator = setup_data_generators(input_shape, batch_size)
        
        # Define models to train
        models = {
            'Basic CNN': create_basic_cnn(input_shape=(*input_shape, 3), num_classes=train_generator.num_classes),
            'ResNet50V2': create_resnet_model(input_shape=(*input_shape, 3), num_classes=train_generator.num_classes),
            'EfficientNetB0': create_efficientnet_model(input_shape=(*input_shape, 3), num_classes=train_generator.num_classes)
        }
        
        # Training settings with improved callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath='models/{epoch:02d}-{val_loss:.2f}.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train all models and collect results
        histories = []
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            try:
                history = model.fit(
                    train_generator,
                    validation_data=val_generator,
                    epochs=30,
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Evaluate model
                val_loss, val_accuracy = model.evaluate(val_generator, verbose=1)
                
                # Save results
                results[name] = {
                    'val_accuracy': float(val_accuracy),
                    'val_loss': float(val_loss),
                    'epochs_trained': len(history.history['loss'])
                }
                
                histories.append(history)
                
                # Save model
                model_filename = f'models/tomato_disease_{name.lower().replace(" ", "_")}_{timestamp}.keras'
                model.save(model_filename)
                print(f"Model saved to {model_filename}")
                
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        # Plot comparison
        if histories:
            plot_comparison(histories, list(models.keys()))
        
        # Save results to JSON
        results_filename = f'results/model_results_{timestamp}.json'
        with open(results_filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_filename}")
        
        # Print summary
        print("\nModel Performance Summary:")
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            print(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")
            print(f"Validation Loss: {metrics['val_loss']:.4f}")
            print(f"Epochs Trained: {metrics['epochs_trained']}")
            
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()