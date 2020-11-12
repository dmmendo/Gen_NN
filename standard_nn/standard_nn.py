import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np
import time
import itertools
import random

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

def output_nn_data(model):
    layer_type_to_idx = get_layer_type_to_idx()
    attr_list,adj_matrix = build_graph_rep(get_layer_metadata(model),layer_type_to_idx)
    f = open("features.txt","a")
    f.write(str(attr_list[0]))
    for i in range(1,len(attr_list)):
        f.write("\t" + str(attr_list[i]))
    f.write("\n")
    f.close()

def run_all_possible_nn(layer_sizes,n_layers):
    nn_iter = itertools.product(layer_sizes,repeat=n_layers)
    max_count = max(int(np.power(len(layer_sizes),n_layers)/50000),1)
    count = int(0)
    for nn in nn_iter:
        if count % max_count == 0:
            model = create_nn_model(nn)

            latency_data = run_nn(model)

            output_latency_data(latency_data)
            output_nn_data(model)
            tf.keras.backend.clear_session()
            exit()
        count = (count + 1) % max_count

def run_test(max_n_layers):
    
    #layer_sizes = create_power2_layer_sizes(n_layer_sizes)
    layer_sizes = fixed_power2_layer_sizes()
    layer_sizes = layer_sizes[::-1]

    n_layers_list = [3,4,5]
    for n_layers in n_layers_list:
        run_all_possible_nn(layer_sizes,n_layers)


run_test(0)
