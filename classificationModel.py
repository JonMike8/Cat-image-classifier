import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import matplotlib.pyplot as plt
from imagePreprocessor import imagePreprocessor

class Classifier: 
    def __init__(self):
        self.IP = imagePreprocessor()
        self.train_ds, self.val_ds, self.class_names, self.num_classes = self.IP.loadData()

        self.AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().prefetch(buffer_size=self.AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=self.AUTOTUNE)

    def createClassifier(self):
#Data augmentation to prevent overfitting, randomly changes images by small increments
        data_aug = keras.Sequential([
            layers.RandomFlip("horizontal", input_shape=(self.IP.img_height, self.IP.img_width, 3)),
            layers.RandomRotation(0.1), 
            layers.RandomZoom(0.1)
        ])
        model = Sequential([
            data_aug,
            layers.Rescaling(1./255, input_shape=(self.IP.img_height, self.IP.img_width, 3)),
            
            # Adding batch normalization
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            
            layers.Dropout(0.2),
            layers.Flatten(),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            
            layers.Dense(self.num_classes)
        ])
        return model
    
    def classifyImage(self, img_path, model): 
        img = tf.keras.utils.load_img(
            img_path, target_size=(self.IP.img_height, self.IP.img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        print("This image is most likely {} with a {:.2f} precent confidence."
              .format(self.class_names[np.argmax(score)], 100*np.max(score)))
        
"""
classifier = Classifier()
model = classifier.createClassifier()
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
model.summary()
history = model.fit(classifier.train_ds, validation_data=classifier.val_ds, epochs=15)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(15)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
model.save("image_classifier.h5")

#Test on unique image
url = "C:\\Users\\jonom\\Desktop\\Coding Projects\\Machine Learning\\Cat or No Cat\\milo.jpg"
classifier.classifyImage(url, model)
"""