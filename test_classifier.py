import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pathlib
import matplotlib.pyplot as plt
from imagePreprocessor import imagePreprocessor
from classificationModel import Classifier


#Test on unique images
url1 = "C:\\Users\\jonom\\Desktop\\Coding Projects\\Machine Learning\\Cat or No Cat\\random_cat1.jpg"
url2 = "C:\\Users\\jonom\\Desktop\\Coding Projects\\Machine Learning\\Cat or No Cat\\random_cat2.jpg"
url3 = "C:\\Users\\jonom\\Desktop\\Coding Projects\\Machine Learning\\Cat or No Cat\\random_cat3.jpg"
url4 = "C:\\Users\\jonom\\Desktop\\Coding Projects\\Machine Learning\\Cat or No Cat\\milo.jpg"
model = load_model("image_classifier.h5")

test_classifier = Classifier()
test_classifier.classifyImage(url1, model)
test_classifier.classifyImage(url2, model)
test_classifier.classifyImage(url3, model)
test_classifier.classifyImage(url4, model)



