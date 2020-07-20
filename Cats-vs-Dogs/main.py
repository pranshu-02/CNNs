#python3

## Libraries
import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile


## Open Data
local_zip = '/tmp/cats-and-dogs.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

## Create Folders For Train And Test Data
try:
    os.mkdir("/tmp/cats-v-dogs")
    os.mkdir("/tmp/cats-v-dogs/training")
    os.mkdir("/tmp/cats-v-dogs/testing")
    os.mkdir("/tmp/cats-v-dogs/training/cats/")
    os.mkdir("/tmp/cats-v-dogs/training/dogs/")
    os.mkdir("/tmp/cats-v-dogs/testing/cats/")
    os.mkdir("/tmp/cats-v-dogs/testing/dogs/")
except OSError:
    print("Error! Cannot Create Folders")


## Split Data To Create Train And Test Set
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    ls= os.listdir(SOURCE)
    dataset=[]
    for img in ls:
        data = SOURCE + img
        if(os.path.getsize(data) > 0):
            dataset.append(img)
        else:
            print('Skipped ' + img)

    random.sample(dataset, len(dataset))
    for x in range(int(SPLIT_SIZE*len(dataset))):
        copyfile(os.path.join(SOURCE,dataset[x]),os.path.join(TRAINING,dataset[x]))
    for x in range(len(dataset)-int((SPLIT_SIZE)*len(dataset))):
        copyfile(os.path.join(SOURCE,dataset[x+int(SPLIT_SIZE*len(dataset))]),os.path.join(TESTING,dataset[x+int(SPLIT_SIZE*len(dataset))]))


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = 0.9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)


## Model: ([Conv]-> [MaxPool])*3 -> [Dense(512)] 
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation="relu",input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation= tf.nn.relu),
    tf.keras.layers.Dense(1,activation= tf.nn.sigmoid)
])


## Create DataGenrators And Data Augmentation For Train And Validation Data
TRAINING_DIR = "/tmp/cats-v-dogs/training"
train_datagen = ImageDataGenerator(rescale= 1/255,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,rotation_range=40,horizontal_flip=True,fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,target_size=(150,150),batch_size=10,class_mode='binary')

VALIDATION_DIR = "/tmp/cats-v-dogs/testing"
validation_datagen = ImageDataGenerator(rescale= 1/255)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,target_size=(150,150),batch_size=10,class_mode='binary')


## Compile And Train Model
model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
history = model.fit(train_generator, epochs=15, verbose=1, validation_data=validation_generator)



