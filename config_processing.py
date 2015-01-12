import numpy as np
import sys
import subprocess
import google.protobuf
import caffe

from paths import *

def conv_layer(h, w, n, group=1, pad_h=0, pad_w=0, stride_h=1, stride_w=1):
    layer = caffe.proto.caffe_pb2.LayerParameter()
    layer.type = caffe.proto.caffe_pb2.LayerParameter.LayerType.Value('CONVOLUTION')
    #layer.convolution_param.engine = caffe.proto.caffe_pb2.ConvolutionParameter.CUDNN
    if (h == w):
        layer.convolution_param.kernel_size = h
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
    while model.layers[k].name != layer_name:
        k += 1
        if (k > len(model.layers)):
            raise IOError('layer with name %s not found' % layer_name)
    return k
 
def accelerate_model(model, layer_to_decompose, rank):
    k = layer_to_decompose
    r = rank
    new_model = caffe.proto.caffe_pb2.NetParameter()
    for i in range(k):
        new_model.layers.extend([model.layers[i]])
    decomposed_layer = model.layers[k]
    if decomposed_layer.type != decomposed_layer.LayerType.Value('CONVOLUTION'):
        raise AttributeError('only convolution layer can be decomposed')
    param = decomposed_layer.convolution_param
    new_model.layers.extend([conv_layer(1, 1, r)])
    new_model.layers.extend([conv_layer(param.kernel_size, 1, r, r, pad_h=param.pad, stride_h=param.stride)])
    new_model.layers.extend([conv_layer(1, param.kernel_size, r, r, pad_w=param.pad, stride_w=param.stride)])
    new_model.layers.extend([conv_layer(1, 1, param.num_output)])
    name = decomposed_layer.name
    for i in range(4):
        new_model.layers[k+i].name = name + '-' + str(i + 1)
        new_model.layers[k+i].bottom.extend([name + '-' + str(i)])
        new_model.layers[k+i].top.extend([name + '-' + str(i + 1)])
    new_model.layers[k].bottom[0] = model.layers[k].bottom[0]
    new_model.layers[k+3].top[0] = model.layers[k].top[0]
    for i in range(k+1, len(model.layers)):
        new_model.layers.extend([model.layers[i]])
    return new_model

def create_deploy_model(model, input_dim=[64, 3, 32, 32]):
    new_model = caffe.proto.caffe_pb2.NetParameter()
    new_model.input.extend(['data'])
    new_model.input_dim.extend(input_dim)
    for i in range(2,len(model.layers)-2):
        new_model.layers.extend([model.layers[i]])
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

    l = ind - 2#layer index in deploy version
    w = net.layers[l].blobs[0].data
    print w.shape
    g = model.layers[ind].convolution_param.group
    if (g > 1):
        weights = np.zeros((w.shape[0], g * w.shape[1], w.shape[2], w.shape[3]))
        for i in range(g):
            weights[i*w.shape[0]/g : (i+1)*w.shape[0]/g, i*w.shape[1] : (i+1)*w.shape[1], :, :] = w[i*w.shape[0]/g:(i+1)*w.shape[0]/g, :, :, :]
            temp = w[i*w.shape[0]/g:(i+1)*w.shape[0]/g, :, :, :]
            np.savetxt('group'+str(i)+'.txt', temp.ravel())
    else:
        weights = w

    bias = net.layers[l].blobs[1]
    np.savetxt('weights.txt', weights.ravel())
    np.savetxt('biases.txt', bias.data.ravel())

    if 1:
        s = weights.shape
        command = 'addpath(\'%s\'); addpath(\'%s\');' % (caffe_root + 'decomp', TENSORLAB_PATH)
        command = command + ' decompose(%d, %d, %d, %d, %d); exit;' % (s[3], s[2], s[1], s[0], R)
        print command
        subprocess.call(['matlab', '-nodesktop', '-nosplash', '-r', command])

    f_x = np.loadtxt('f_x.txt').transpose()
    f_y = np.loadtxt('f_y.txt').transpose()
    f_c = np.loadtxt('f_c.txt').transpose()
    f_n = np.loadtxt('f_n.txt')
    if 0:    
        f_x = np.random.standard_normal(f_x.shape)
        f_y = np.random.standard_normal(f_y.shape)
        f_c = np.random.standard_normal(f_c.shape)
        f_n = np.random.standard_normal(f_n.shape)

    n = model.layers[ind].convolution_param.num_output
    d = model.layers[ind].convolution_param.kernel_size
    c = weights.shape[1]# / model.layers[ind].convolution_param.group #i don't know what i'm doing 
    
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
