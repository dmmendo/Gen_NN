import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
import time
import itertools
import random

def create_nn_model(nn_layers):
    input_l = Input(shape=(int(nn_layers[0])))
    nn_l = Dense(nn_layers[1], activation="relu")(input_l)
    for i in range(2,len(nn_layers)):
        nn_l = Dense(nn_layers[i], activation="relu")(nn_l)
    model = Model(inputs=input_l, outputs=nn_l)
    return model

def create_power2_layer_sizes(num_sizes):
    layer_sizes = np.power(2,[i for i in range(num_sizes)])
    return layer_sizes

def fixed_power2_layer_sizes():
    layer_sizes = np.power(2,[0,5,6,7,8,9,10,11,12,13,14])
    return layer_sizes

def run_nn(model):
    latency_data = []
    for i in range(20):
        rand_ex = np.random.randn(1,model.layers[0].input_shape[0][1]) * 0.01
        start_t = time.time()
        model.predict(rand_ex)
        latency_data.append(time.time() - start_t)
    return latency_data

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

def run_all_possible_nn(layer_sizes,n_layers):
    nn_iter = itertools.product(layer_sizes,repeat=n_layers)
    max_count = int(np.power(len(layer_sizes),n_layers)/50000)
    count = int(0)
    for nn in nn_iter:
        if count % max_count == 0:
            model = create_nn_model(nn)

            latency_data = run_nn(model)

            output_latency_data(latency_data)
            output_nn_data(nn)
            tf.keras.backend.clear_session()
        count = (count + 1) % max_count

def run_test(max_n_layers):
    
    #layer_sizes = create_power2_layer_sizes(n_layer_sizes)
    layer_sizes = fixed_power2_layer_sizes()
    layer_sizes = layer_sizes[::-1]

    n_layers_list = [9,8,7]
    for n_layers in n_layers_list:
        run_all_possible_nn(layer_sizes,n_layers)


run_test(0)
