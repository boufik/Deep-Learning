import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# Set Matplotlib defaults
plt.style.use('seaborn-whitegrid')
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)

# 1. Read and print the dataset
concrete = pd.read_csv('../input/dl-course-data/concrete.csv')
print(concrete.head(), "Shape = {}".format(concrete.shape), sep='\n\n', end='\n\n')

# 2. Create 3 hidden layers and 1 output layer
input_shape = [8]
hidden1 = layers.Dense(units=512, activation='relu', input_shape=input_shape)
hidden2 = layers.Dense(units=512, activation='relu')
hidden3 = layers.Dense(units=512, activation='relu')
output = layers.Dense(units=1)
model = keras.Sequential([hidden1, hidden2, hidden3, output])

# 3. Rewrite layers
model = keras.Sequential([ layers.Dense(units=32, input_shape=[8]), layers.Activation('relu'),
    					layers.Dense(units=32), layers.Activation('relu'),
   						layers.Dense(units=1) ])

# 4. Alternatives to ReLU
choices = ['relu', 'elu', 'selu', 'swish']

activation_layer = layers.Activation('swish')
x = tf.linspace(-3.0, 3.0, 100)
y = activation_layer(x) # once created, a layer is callable just like a function

plt.figure(dpi=100)
plt.plot(x, y)
plt.xlim(-3, 3)
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()
