import numpy as np
import sys
import subprocess
import google.protobuf
import caffe

from sktensor import dtensor, cp_als

from paths import *

def conv_layer(h, w, n, group=1, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    layer = caffe.proto.caffe_pb2.LayerParameter()
    layer.type = 'Convolution'
    if (h == w):
        layer.convolution_param.kernel_size.append(h)
    else:
        layer.convolution_param.kernel_h = h
        layer.convolution_param.kernel_w = w
    layer.convolution_param.num_output = n
    if (group != 1):
        layer.convolution_param.group = group
    if (pad_h != 0 or pad_w != 0):
        layer.convolution_param.pad_h = pad_h
        layer.convolution_param.pad_w = pad_w
    if (stride_h != 1 or stride_w != 1):
        layer.convolution_param.stride_h = stride_h
        layer.convolution_param.stride_w = stride_w
    return layer

def find_layer_by_name(model, layer_name):
    k = 0
    while model.layer[k].name != layer_name:
        k += 1
        if (k > len(model.layer)):
            raise IOError('layer with name %s not found' % layer_name)
    return k
 
def accelerate_model(model, layer_to_decompose, rank):
    k = layer_to_decompose
    r = rank
    new_model = caffe.proto.caffe_pb2.NetParameter()
    for i in range(k):
        new_model.layer.extend([model.layer[i]])
    decomposed_layer = model.layer[k]
    if decomposed_layer.type != 'Convolution':
        raise AttributeError('only convolution layer can be decomposed')
    param = decomposed_layer.convolution_param   
    if not hasattr(param, 'pad'):
        param.pad = [0]
    if param.pad == []:
        param.pad.append(0)
    if not hasattr(param, 'stride'):
        param.stride = [1]
    new_model.layer.extend([conv_layer(1, 1, r)])
    new_model.layer.extend([conv_layer(param.kernel_size[0], 1, r, r, pad_h=param.pad[0], stride_h=param.stride[0])])
    new_model.layer.extend([conv_layer(1, param.kernel_size[0], r, r, pad_w=param.pad[0], stride_w=param.stride[0])])
    new_model.layer.extend([conv_layer(1, 1, param.num_output)])
    name = decomposed_layer.name
    for i in range(4):
        new_model.layer[k+i].name = name + '-' + str(i + 1)
        new_model.layer[k+i].bottom.extend([name + '-' + str(i)])
        new_model.layer[k+i].top.extend([name + '-' + str(i + 1)])
    new_model.layer[k].bottom[0] = model.layer[k].bottom[0]
    new_model.layer[k+3].top[0] = model.layer[k].top[0]
    for i in range(k+1, len(model.layer)):
        new_model.layer.extend([model.layer[i]])
    return new_model

def create_deploy_model(model, input_dim=[64, 3, 32, 32]):
    new_model = caffe.proto.caffe_pb2.NetParameter()
    new_model.input.extend(['data'])
    new_model.input_dim.extend(input_dim)
    for i in range(2,len(model.layer)-2):
        new_model.layer.extend([model.layer[i]])
    return new_model
    
def load_model(filename):
    model = caffe.proto.caffe_pb2.NetParameter()
    input_file = open(filename, 'r')
    google.protobuf.text_format.Merge(str(input_file.read()), model)
    input_file.close()
    return model

def save_model(model, filename):
    output_file = open(filename, 'w')
    google.protobuf.text_format.PrintMessage(model, output_file)
    output_file.close()

def prepare_models(LAYER, R, NET_PATH, NET_NAME, INPUT_DIM):
    PATH = NET_PATH
    NET_PREFIX = PATH + NET_NAME
    input_dim = INPUT_DIM
    
    model = load_model(NET_PREFIX + '.prototxt')
    ind = find_layer_by_name(model, LAYER)
    new_model = accelerate_model(model, ind, R)
    save_model(new_model, NET_PREFIX + '_accelerated.prototxt')
    new_deploy = create_deploy_model(new_model, input_dim)
    save_model(new_deploy, NET_PREFIX + '_accelerated_deploy.prototxt')
    deploy = create_deploy_model(model, input_dim)
    save_model(deploy, NET_PREFIX + '_deploy.prototxt')

    net = caffe.Classifier(NET_PREFIX + '_deploy.prototxt', NET_PREFIX + '.caffemodel')
    fast_net = caffe.Classifier(NET_PREFIX + '_accelerated_deploy.prototxt', NET_PREFIX + '.caffemodel')

    l = ind - 1#layer index in deploy version
    weights = net.layers[l].blobs[0].data
    bias = net.layers[l].blobs[1]

    T = dtensor(weights)
    P, fit, itr, exectimes = cp_als(T, R, init='random')
    f_x = (np.array(P.U[3])*(P.lmbda)).T
    f_y = np.array(P.U[2]).T
    f_c = np.array(P.U[1]).T
    f_n = np.array(P.U[0]) 

    n = model.layer[ind].convolution_param.num_output
    d = model.layer[ind].convolution_param.kernel_size[0]
    c = weights.shape[1]
    
    f_y = np.reshape(f_y, [R, 1, d, 1])
    f_x = np.reshape(f_x, [R, 1, 1, d])
    f_c = np.reshape(f_c, [R, c, 1, 1])
    f_n = np.reshape(f_n, [n, R, 1, 1])

    np.copyto(fast_net.layers[l].blobs[0].data, f_c)
    np.copyto(fast_net.layers[l+1].blobs[0].data, f_y)
    np.copyto(fast_net.layers[l+2].blobs[0].data, f_x)
    np.copyto(fast_net.layers[l+3].blobs[0].data, f_n)
    np.copyto(fast_net.layers[l+3].blobs[1].data, bias.data)

    fast_net.save(NET_PREFIX + '_accelerated.caffemodel')
