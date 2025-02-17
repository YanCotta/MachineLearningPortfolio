"""
Data loading and preprocessing module for the CNN Image Classifier.

This module handles all data-related operations including loading, augmentation,
and preprocessing of image data for both training and inference.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles data loading and preprocessing operations."""

    def __init__(
        self,
        img_size: Tuple[int, int] = (64, 64),
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """
        Initialize the DataLoader.

        Args:
            img_size: Tuple of (height, width) for input images
            batch_size: Number of samples per batch
            validation_split: Fraction of data to use for validation
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # Initialize data generators with augmentation for training
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2
        )

        # Only rescaling for validation/test data
        self.test_datagen = ImageDataGenerator(rescale=1./255)

    def load_data(
        self,
        train_path: str,
        test_path: str
    ) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator,
               tf.keras.preprocessing.image.DirectoryIterator]:
        """
        Load and prepare training and test datasets.

        Args:
            train_path: Path to training data directory
            test_path: Path to test data directory

        Returns:
            Tuple of (training_set, test_set) DirectoryIterators
        """
        training_set = self.train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        test_set = self.test_datagen.flow_from_directory(
            test_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary'
        )

        logger.info(f"Loaded {training_set.samples} training samples")
        logger.info(f"Loaded {test_set.samples} test samples")

        return training_set, test_set

    def prepare_single_image(self, image_path: str) -> np.ndarray:
        """
        Prepare a single image for inference.

        Args:
            image_path: Path to the image file

        Returns:
            Preprocessed image array ready for model prediction
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        img = image.load_img(image_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0  # Rescale to match training data

    @staticmethod
    def get_class_names(dataset: tf.keras.preprocessing.image.DirectoryIterator) -> dict:
        """
        Get the mapping of class indices to class names.

        Args:
            dataset: DirectoryIterator containing the class information

        Returns:
            Dictionary mapping class indices to class names
        """
        return dataset.class_indices