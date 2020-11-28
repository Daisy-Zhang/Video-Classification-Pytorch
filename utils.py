import sys
import os
import numpy as np
import cv2
import os
import json
import math
import glob
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

from extract_features import getExtractor, extractFeatures
import conf

def getModel(model_type, use_gpu):
    if model_type == 'RNN':
        from models.RNN import rnn
        model = rnn()
    elif model_type == 'LRCN':
        from models.LRCN import lrcn
        model = lrcn()
    elif model_type == 'ALSTM':
        from models.ALSTM import alstm
        model = alstm()
    elif model_type == 'CNNLSTM':
        from models.CNNLSTM import cnnLstm
        model = cnnLstm(num_class = 10)
    else:
        print('this model is not supported')
        sys.exit()
    
    if use_gpu:
        model = model.cuda()
    
    return model

def myEval(model, data_path, use_gpu, image_size, cnn_type, seq_length):
    model.eval()

    for video_name in os.listdir(data_path):
        if video_name.endswith('.mp4') or video_name.endswith('.avi'):
            #continue
            video_path = data_path + '/' + video_name
            #print(video_path)
            path = os.path.join(data_path, video_name[0 : len(video_name) - 4] + '-' + str(seq_length) + '-features-' + cnn_type)
            
            if not os.path.isfile(path + '.npy'):
                frames_path = sorted(glob.glob(os.path.join(data_path, video_name[0 : len(video_name) - 4] + '*jpg')))
                #print(frames_path)
                sequence = list()

                extractor = getExtractor(cnn_type, use_gpu)
                for image_path in frames_path:
                    # 可以在此处加采样间隔
                    features = extractFeatures(extractor, image_path, use_gpu, image_size)
                    sequence.append(features)

                # Save the sequence.
                np.save(path, sequence)

            features = np.load(path + '.npy')
            features = torch.tensor(features)
            features = Variable(features)
            features = features.unsqueeze(0)
            
            if use_gpu:
                features = features.cuda()
            print(features.size())
            outputs = model(features)
            #print(outputs)

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]