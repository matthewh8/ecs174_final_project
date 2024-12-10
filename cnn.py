# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
import os

# Set to load truncated images without failing
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set the path to your dataset
data_dir = "/Users/matthew/Documents/ecs174/final_project/new_data"

# Step 1: Define the CNN model with input layer as per Keras recommendation
def create_cnn(input_shape=(128, 128, 3), num_classes=4):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 2: Load and prepare the dataset
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
    os.path.join(data_dir, 'train'),
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(data_dir, 'valid'),
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical'
)

# Step 3: Train the CNN model
model = create_cnn(input_shape=(128, 128, 3), num_classes=train_generator.num_classes)

# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stopping]
)

# Step 4: Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Step 5: Plot accuracy and loss
def plot_training_history(history):
    plt.figure(figsize=(14, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Step 6: Display initial images from the dataset
def display_initial_images(data_generator, num_images=5):
    x_batch, y_batch = next(data_generator)  # Get a batch of images and labels
    plt.figure(figsize=(12, 6))
    
    for i in range(num_images):
        ax = plt.subplot(1, num_images, i + 1)
        plt.imshow(x_batch[i])
        plt.title("Class: {}".format(y_batch[i].argmax()))  # Display the class of the image
        plt.axis("off")
    
    plt.show()

display_initial_images(train_generator)

# Save the model in Keras format
model.save('tomato_leaf_disease_simple_cnn.keras')
