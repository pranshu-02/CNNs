#python3

## Libraries
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## Load Data
def get_data(filename):
    with open(filename) as training_file:
        data = np.loadtxt(training_file, delimiter=',', skiprows=1)
        labels = data[:, 0].astype(int)-1
        images = data[:, 1:].astype(float).reshape((data.shape[0], 28, 28))
        data = None
    return images, labels


training_images, training_labels = get_data('data/sign_mnist_train.csv')
testing_images, testing_labels = get_data('data/sign_mnist_test.csv')


## Preprocess Data
training_images = np.expand_dims(training_images,-1)
testing_images = np.expand_dims(testing_images,-1)
training_labels = tf.keras.utils.to_categorical(training_labels, 24)
testing_labels = tf.keras.utils.to_categorical(testing_labels, 24)


## Create DataGenrators And Data Augmentation For Train And Validation Data
train_datagen = ImageDataGenerator(rescale=1/255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1/255)
train_datagen.fit(training_images)
validation_datagen.fit(testing_images)

## Model: ([Conv]-> [MaxPool])*3 -> [Dense(512)]
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(24, activation='softmax')
])

## Compile And Train Model 
model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit_generator(train_datagen.flow(training_images,training_labels),epochs=3, validation_data = validation_datagen.flow(testing_images,testing_labels))

## Evaluate Model
model.evaluate(testing_images, testing_labels)


