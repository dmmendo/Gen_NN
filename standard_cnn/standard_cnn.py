from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import time
import random
from itertools import product

def plain_cnn_layer(input,filter,stride):
  output = Activation(activation='relu')(BatchNormalization()(Conv2D(filters=filter,kernel_size=(3,3),strides=(stride,stride),padding="same")(input)))
  return output

def gen_std_CNN(filter_counts,layer_counts):
  inputs = Input(shape=(224,224,3))
  x = inputs
  network = []
  for stage in range(0,5):
    num_layers = layer_counts[stage]
    for i in range(0,num_layers-1):
      x = plain_cnn_layer(x,filter_counts[stage],1)
      network.append(filter_counts[stage])
      network.append(1)
    x = plain_cnn_layer(x,filter_counts[stage+1],2)
    network.append(filter_counts[stage+1])
    network.append(2)
  model = Model(inputs, x)
  return model,network

def output_latency_data(latency_data):
    fileout_name = "labels.txt"
    f = open("labels.txt","a")
    for percent in [50,60,70,80,90]:
        f.write(str(np.percentile(latency_data,percent)) + "\t")
    f.write(str(max(latency_data)) + "\n")
    f.close()

def output_nn_data(network):
    f = open("features.txt","a")
    f.write(str(network[0]))
    for i in range(1,len(network)):
        f.write("\t" + str(network[i]))
    f.write("\n")
    f.close()

def run_nn(model):
    latency_data = []
    for i in range(20):
        rand_ex = np.random.randn(1,224,224,3)
        start_t = time.time()
        model.predict(rand_ex)
        latency_data.append(time.time() - start_t)
    return latency_data

def run_test():
  filter_counts = [np.power(2,i) for i in range(6,12)]

  for entry in product(range(1,11),repeat=5):
    model,network = gen_std_CNN(filter_counts,entry)
    latency_data = run_nn(model)
    #model.summary()
    tf.keras.backend.clear_session()
    output_latency_data(latency_data)
    output_nn_data(network)

run_test()
