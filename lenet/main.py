import numpy as np
import sys, os, subprocess

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))                                                                        
from paths import caffe_root
sys.path.append(caffe_root + 'python')

import caffe
from config_processing import *

LAYER = 'conv1'
R = 4
NET_PATH = 'lenet/'
NET_NAME = 'lenet'
INPUT_DIM = [64, 1, 28, 28]

prepare_models(LAYER, R, NET_PATH, NET_NAME, INPUT_DIM)
