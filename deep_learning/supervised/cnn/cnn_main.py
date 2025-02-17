"""
Convolutional Neural Network (CNN) for Binary Image Classification
This script implements a CNN to classify images into two categories (cats and dogs).
The dataset is quite large (over 10,000 pictures) and is not included in this repository, 
but I've added it to a Google Drive in order for you to download,
you can find it here: https://drive.google.com/file/d/1vtO7vdRndB0fCsz6RYCgH6DiYWLY7GUG/view?usp=sharing
"""

import tensorflow as tf
import numpy as np
import logging
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageClassificationCNN:
    """A CNN model for binary image classification."""
    
    def __init__(self, img_size: Tuple[int, int] = (64, 64), batch_size: int = 32):
        """
        Initialize the CNN model.
        
        Args:
            img_size: Tuple of (height, width) for input images
            batch_size: Batch size for training
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None

    def prepare_data(self, train_path: str, test_path: str) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator,
                                                                    tf.keras.preprocessing.image.DirectoryIterator]:
        """
        Prepare and augment training and test data.
        
        Args:
            train_path: Path to training data directory
            test_path: Path to test data directory
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        training_set = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        test_set = test_datagen.flow_from_directory(
            test_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        logger.info(f"Found {training_set.samples} training samples")
        logger.info(f"Found {test_set.samples} test samples")

        return training_set, test_set

    def build_model(self):
        """Build the CNN architecture."""
        self.model = tf.keras.models.Sequential([
            # First Convolutional Layer
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=[*self.img_size, 3]),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            
            # Second Convolutional Layer
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            
            # Third Convolutional Layer
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
            
            # Flatten and Dense Layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        
        self.model.summary()

    def train(self, training_set, test_set, epochs: int = 25):
        """Train the model and plot training history."""
        self.history = self.model.fit(
            training_set,
            validation_data=test_set,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
            ]
        )
        
        self._plot_training_history()

    def predict_image(self, image_path: str) -> str:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            str: Prediction result ('cat' or 'dog')
        """
        try:
            test_image = image.load_img(image_path, target_size=self.img_size)
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            
            result = self.model.predict(test_image)
            prediction = 'dog' if result[0][0] > 0.5 else 'cat'
            confidence = result[0][0] if result[0][0] > 0.5 else 1 - result[0][0]
            
            logger.info(f"Prediction: {prediction} (Confidence: {confidence:.2%})")
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def _plot_training_history(self):
        """Plot training history metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'])
        ax1.plot(self.history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend(['Train', 'Validation'])
        
        # Plot loss
        ax2.plot(self.history.history['loss'])
        ax2.plot(self.history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend(['Train', 'Validation'])
        
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function."""
    try:
        # Initialize model
        cnn = ImageClassificationCNN()
        
        # Prepare data
        training_set, test_set = cnn.prepare_data(
            'dataset/training_set',
            'dataset/test_set'
        )
        
        # Build and train model
        cnn.build_model()
        cnn.train(training_set, test_set)
        
        # Make a prediction
        prediction = cnn.predict_image('dataset/single_prediction/cat_or_dog_1.jpg')
        print(f"Final prediction: {prediction}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()