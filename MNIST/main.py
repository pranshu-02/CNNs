#python3

## Libraries
import tensorflow as tf

## Callback To Stop At 99.8% Accuracy
class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs={}):
            if(logs.get('accuracy') > 0.998 ):
                print( "Reached 99.8% accuracy so cancelling training!")
                self.model.stop_training=True

## Load Data
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

## Preprocessing Data
training_images=training_images.reshape((60000,28,28,1))
training_images=training_images/255.0
test_images=test_images.reshape((10000,28,28,1))
test_images=test_images/255.0
callback=myCallback()


## Model : [Conv]-> [MaxPool]-> [Dense(128)]
model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
            tf.keras.layers.MaxPool2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128,activation= tf.nn.relu),
            tf.keras.layers.Dense(10,activation= tf.nn.softmax)
])

## Compile and Train Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit( training_images,training_labels,epochs=20,callbacks=[callback])
