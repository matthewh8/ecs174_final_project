import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Input, 
    Dropout, BatchNormalization, Flatten
)
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import kaggle
import os
import json
from datetime import datetime

def download_dataset():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'cookiefinder/tomato-disease-multiple-sources',
        path='.',
        unzip=True
    )
    print("Dataset downloaded and extracted successfully")

def create_basic_cnn(input_shape=(128, 128, 3), num_classes=4):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def create_resnet_model(input_shape=(128, 128, 3), num_classes=4):
    base_model = ResNet50V2(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = False  # Freeze base model
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs, outputs)

def create_efficientnet_model(input_shape=(128, 128, 3), num_classes=4):
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

def plot_comparison(histories, model_names):
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
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def main():
    # Download dataset if needed
    if not os.path.exists('train'):
        download_dataset()
    
    # Data preparation
    batch_size = 32
    input_shape = (128, 128)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        'valid',
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Define models to train
    models = {
        'Basic CNN': create_basic_cnn(input_shape=(128, 128, 3), num_classes=train_generator.num_classes),
        'ResNet50V2': create_resnet_model(input_shape=(128, 128, 3), num_classes=train_generator.num_classes),
        'EfficientNetB0': create_efficientnet_model(input_shape=(128, 128, 3), num_classes=train_generator.num_classes)
    }
    
    # Training settings
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train all models and collect results
    histories = []
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=30,
            callbacks=[early_stopping]
        )
        
        # Evaluate model
        val_loss, val_accuracy = model.evaluate(val_generator)
        
        # Save results
        results[name] = {
            'val_accuracy': float(val_accuracy),
            'val_loss': float(val_loss),
            'epochs_trained': len(history.history['loss'])
        }
        
        histories.append(history)
        
        # Save model
        model.save(f'tomato_disease_{name.lower().replace(" ", "_")}.keras')
    
    # Plot comparison
    plot_comparison(histories, list(models.keys()))
    
    # Save results to JSON
    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\nModel Performance Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")
        print(f"Validation Loss: {metrics['val_loss']:.4f}")
        print(f"Epochs Trained: {metrics['epochs_trained']}")

if __name__ == "__main__":
    main()