import sys
import numpy as np
import matplotlib.pyplot as plt
import collections
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import random
import time



gamma = 0.95
n = 1000                            
w_min, w_max = 2, 5
w_default = np.linspace(w_min, w_max, n+1)  


def u(c, σ=1.3):
    return (c**(1 - σ) - 1) / (1 - σ)


l_ = []
for salary in w_default:
    l_.append(u(salary)/(1 - gamma))
    
l_35 = []
for i, w in enumerate(w_default):
    if w<=3.5:
        l_35.append(l_[i])
    else:
        l_35.append(19.2)
        
        


def create_model():
    inputs = layers.Input(shape=(1,))
    
    layer1 = layers.Dense(16, activation="linear")(inputs)
    layer2 = layers.Dense(16, activation="sigmoid")(layer1)
    layer3 = layers.Dense(16, activation="sigmoid")(layer2)
    

    action = layers.Dense(1, activation="linear")(layer3)

    return keras.Model(inputs=inputs, outputs=action)



model = create_model()
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_function = keras.losses.MeanSquaredError()
model.compile(optimizer, loss_function)

start = time.time()
with tf.device("/gpu:0"):
    hist = model.fit(w_default, np.array(l_35),  epochs = 2000, verbose = 0)

end = time.time()

sys.stdout.write('Temps GPU:')
sys.stdout.write(str(end-start))
sys.stdout.write('Loss :')
sys.stdout.write(str(hist.history['loss'][-1]))


model = create_model()
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_function = keras.losses.MeanSquaredError()
model.compile(optimizer, loss_function)

start = time.time()
with tf.device("/cpu:0"):
    hist = model.fit(w_default, np.array(l_35),  epochs = 2000, verbose = 0)

end = time.time()

sys.stdout.write('Temps CPU :')
sys.stdout.write(str(end-start))
sys.stdout.write('Loss :')
sys.stdout.write(str(hist.history['loss'][-1]))

