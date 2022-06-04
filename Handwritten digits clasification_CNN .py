import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.optimizers import adam_v2
from tensorflow.keras.utils import to_categorical

# loading the Data 
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(60000, 28, 28, 1)         
test_images = test_images.reshape(10000, 28, 28, 1)


# Applying CNN 

# Building the model
model = Sequential() 

model.add(Conv2D(filters = 32, kernel_size = (4, 4), strides=(2, 2), input_shape=(28, 28, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))) 

model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = (4, 4), strides=(2, 2)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())

model.add(Dense(10))
model.add(Activation("softmax"))

# Training the model
opt = adam_v2.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

model.fit(train_images, to_categorical(train_labels), batch_size = 64, epochs =11,
        validation_data=(test_images, to_categorical(test_labels)))

# Testing and model results 
prediction = model.predict(test_images)

evaluation = model.evaluate(test_images, to_categorical(test_labels), verbose=1)

print ("Prediction: \n", prediction)
print("Evaluation: \n",evaluation)
print(model.summary())
