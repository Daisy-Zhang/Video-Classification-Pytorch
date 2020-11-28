import argparse
from matplotlib import pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import cv2
import os
import json
import math

from utils import getModel, WarmUpLR, myEval
import conf

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True, help='model type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action="store_true", help = 'use gpu or not')
    parser.add_argument('-data_path', type=str, required=True, help='test data path')
    parser.add_argument('-image_size', type=int, required=True, help='input image size')
    parser.add_argument('-cnn_type', type=str, required=True, help='lstm feature extractor type')
    parser.add_argument('-seq_length', type = int, required = True, help = 'sequences length')
    args = parser.parse_args()

    model = getModel(model_type = args.model, use_gpu = args.gpu)
    model.load_state_dict(torch.load(args.weights), args.gpu)
    
    myEval(model, args.data_path, args.gpu, args.image_size, args.cnn_type, args.seq_length)