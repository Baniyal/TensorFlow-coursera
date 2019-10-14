import tensorflow as tf
import numpy as np
from tensorflow import keras


#this next line is a neural network.It has 1 layer and that 1 layer has 1 neuron
#and the input shape to it is just 1 value
model = tf.keras.Sequential([keras.layers.Dense(units = 1 ,input_shape = [1])])
#now inorder to compile our neural network,we specify two functions
#a loss and an optimizer.Loss function measures the guessed answer against the known
#correct answers and measure how well or badly it did
#Optimizer function is used to make another guess,based on how the loss function went
#it will try to minimize that loss(error) 
model.compile(optimizer = "sgd" , loss = "mean_squared_error")
#here sgd is stochastic gradient descent
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
#now we train the neural network on the given data
model.fit(xs, ys , epochs = 1000)
#epochs tells us how many times it will loop over
model.predict([10.0])