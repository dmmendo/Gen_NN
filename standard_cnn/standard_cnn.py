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

def get_layer_type_to_idx():
  layer_to_idx = {'InputLayer':0, 'DepthwiseConv2D': 1, 'SeparableConv2D':1, 'Conv2D':1, 'Add':2, 'Concatenate':2, 'Dense':3}
  return layer_to_idx

def get_layer_metadata(model):
  metadata = []
  for layer in model.layers:
    layer_features = {}
    layer_features['name'] = layer.output.name
    layer_features['count_params'] = layer.count_params()
    layer_features['type'] = layer.__class__.__name__
    #print(layer.output.name,layer.count_params())
    if type(layer.input) == type(list()):
      #print("\t" + str([layer.input[i].name for i in range(len(layer.input))]))
      layer_features['input'] = [layer.input[i].name for i in range(len(layer.input))]
    else:
      #print("\t" + layer.input.name)
      layer_features['input'] = [layer.input.name]
    metadata.append(layer_features)
  return metadata

def build_graph_rep(metadata,layer_type_to_idx):
  attribute_list = []
  name_to_idx = {}
  adj_size = 0
  for i in range(len(metadata)):
    layer_features = metadata[i]
    layer_type = layer_features['type']
    if layer_type in layer_type_to_idx:
      adj_size += 1
  adj_matrix = np.zeros((adj_size,adj_size))
  idx = 0
  for i in range(0, len(metadata)):
    layer_features = metadata[i]
    name = layer_features['name']
    count_params = layer_features['count_params']
    layer_type = layer_features['type']
    layer_inputs = layer_features['input']

    if layer_type in layer_type_to_idx:
      layer_type_idx = layer_type_to_idx[layer_type]
    else:
      for entry in layer_inputs:
        if entry != name:
          name_to_idx[name] = name_to_idx[entry]
          break
      continue

    name_to_idx[name] = idx

    #fill in adjacency matrix
    for entry in layer_inputs:
      if entry != name:
        input_idx = name_to_idx[entry]
        adj_matrix[input_idx][idx] = 1   
    
    #append attribute list
    #attribute_list.append((layer_type_idx,count_params))
    attribute_list.append(layer_type_idx)
    attribute_list.append(count_params)



    idx += 1

  #post process attribute list
  #max_layer_type_idx = max([layer_type_to_idx[entry] for entry in layer_type_to_idx])
  #for i in range(len(attribute_list)):
  #  layer_type_idx = attribute_list[i][0]
  #  count_params = attribute_list[i][1]
  #  new_entry = np.array([count_params])
  #  attribute_list[i] = new_entry
  attribute_list = np.array(attribute_list)
  return attribute_list,adj_matrix

def flatten_graph(attr_list,adj_matrix):
    attr_list = attr_list.reshape((attr_list.shape[0]*attr_list.shape[1],1))
    adj_matrix = adj_matrix.reshape((adj_matrix.shape[0]*adj_matrix.shape[1],1))
    return np.concatenate((attr_list,adj_matrix))


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

def output_nn_data(model):
    layer_type_to_idx = get_layer_type_to_idx()
    attr_list,adj_matrix = build_graph_rep(get_layer_metadata(model),layer_type_to_idx)
    f = open("features.txt","a")
    f.write(str(attr_list[0]))
    for i in range(1,len(attr_list)):
        f.write("\t" + str(attr_list[i]))
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
    output_nn_data(model)

run_test()
