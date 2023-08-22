import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
import pathlib
import matplotlib.pyplot as plt

class imagePreprocessor:
    batch_size = 64
    img_height = 180
    img_width = 180
    data_dir = "C:\\Users\\jonom\\Desktop\\Coding Projects\\Machine Learning\\Cat or No Cat\\data"
   
    def numImages(self):
        data_path = pathlib.Path(self.data_dir)
        cat_images = len(list(data_path.glob("training_set/cats/*.jpg"))) + len(list(data_path.glob("test_set/cats/*.jpg")))
        notcat_images = len(list(data_path.glob("training_set/notcats/*.jpg"))) + len(list(data_path.glob("test_set/notcats/*.jpg")))
        total_images = cat_images + notcat_images
        return f"Total: {total_images} Cats: {cat_images} NotCats: {notcat_images}"
    
    def loadData(self): 
        train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir+"\\training_set",
            validation_split=0.2, 
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
            )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir+"\\test_set",
            validation_split=0.2, 
            subset="training",
            seed=123,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
            )
        class_names = train_ds.class_names
        num_classes = len(class_names)
    
        return train_ds, val_ds, class_names, num_classes
""" 
IP = imagePreprocessor()
num_images = IP.numImages()
print(num_images)

train_ds, val_ds, class_names, num_classes = IP.loadData()
for image_batch, label_batch in train_ds:
    # 'image_batch' contains a batch of images
    # 'label_batch' contains corresponding label indices
    print("Image batch shape:", image_batch.shape)
    print("Label batch shape:", label_batch.shape)
    break  # Break after processing one batch for demonstration purposes
"""

