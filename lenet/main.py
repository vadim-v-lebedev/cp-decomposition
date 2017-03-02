import numpy as np
import sys, os, subprocess
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))                                                                        
from paths import caffe_root
sys.path.append(caffe_root + 'python')

import caffe
from config_processing import *

parser = argparse.ArgumentParser()
parser.add_argument('R', type=int, help='number of components in the decomposition')
parser.add_argument('layer', type=str, nargs='?', help='which conv layer to decompose', default='conv1')
args = parser.parse_args()

LAYER = args.layer
R = args.R
NET_PATH = 'lenet/'
NET_NAME = 'lenet'
INPUT_DIM = [64, 1, 28, 28]

prepare_models(LAYER, R, NET_PATH, NET_NAME, INPUT_DIM)
