import numpy as np
import os.path
import argparse

import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms

from PIL import Image
import glob

# 本地环境ssl证书问题
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 暂且使用等距采样
img_to_tensor = transforms.ToTensor()

def getExtractor(model_type, use_gpu):
    if model_type == 'vgg16':
        model=models.vgg16(pretrained=True)
        model=model.eval()
    else:
        print('this model is not supported')
        sys.exit()
    
    if use_gpu:
        model.cuda()

    return model

def getFramesPath(data_dir, train_or_test, fake_or_real, video_name):
    path = os.path.join(data_dir, train_or_test, fake_or_real)
    #print(path)
    images = sorted(glob.glob(os.path.join(path, video_name[0 : len(video_name) - 4] + '*jpg')))
    
    return images

def extractFeatures(model, image_path, use_gpu, image_size):
    img = Image.open(image_path)
    #print(img.size)
    img = img.resize((image_size, image_size))
    #print(img.size)
    tensor = img_to_tensor(img)
    #print(tensor.size())

    if use_gpu:
        tensor = tensor.cuda()
    
    # 神经网络要求：数量*通道数*长*宽
    # 而处理单张图片只有：通道数*长*宽
    tensor = tensor.unsqueeze(0)
    #print(tensor.size())

    result = model(Variable(tensor))
    result_npy = result.data.cpu().numpy()
    #print(result_npy.size)
    #print(result_npy[0].size)
    
    return result_npy[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type = str, required = True, help = 'extractor model type')
    parser.add_argument('-seq_length', type = int, required = True, help = 'sample frames length')
    parser.add_argument('-image_size', type = int, required = True, help = 'crop image size')
    parser.add_argument('-sample_frames', type = int, required = True, help = 'sampled frames')
    parser.add_argument('-data_dir', type = str, required = True, help = 'whole image folder dir')
    parser.add_argument('-gpu', action="store_true", help = 'use gpu or not')
    args = parser.parse_args()
    
    model_type = args.model
    seq_length = args.seq_length
    data_dir = args.data_dir
    image_size = args.image_size
    sample_frames = args.sample_frames

    # get the extractor.
    model = getExtractor(model_type, args.gpu)

    # Loop through data.
    for train_or_test in os.listdir(data_dir):
        if train_or_test != 'train' and train_or_test != 'test':
            continue
        f = open(data_dir + '/sequences/' + str(train_or_test) + '_metadata.txt', 'w')
        print(train_or_test)
        for fake_or_real in os.listdir(data_dir + '/' + train_or_test):
            for video_name in os.listdir(data_dir + '/' + train_or_test + '/' + fake_or_real):
                if not video_name.endswith('.mp4'):
                    continue
                if fake_or_real == 'fake':
                    f.write(video_name + ' 1\n')
                else:
                    f.write(video_name + ' 0\n')
                
                path = os.path.join(data_dir, 'sequences', video_name[0 : len(video_name) - 4] + '-' + str(seq_length) + '-features-' + model_type)
                
                if os.path.isfile(path + '.npy'):
                    continue
                
                frames_path = getFramesPath(data_dir, train_or_test, fake_or_real, video_name)
                #print(frames_path)
                sequence = list()

                for image_path in frames_path:
                    # 可以在此处加采样间隔
                    features = extractFeatures(model, image_path, args.gpu, image_size)
                    sequence.append(features)

                # Save the sequence.
                np.save(path, sequence)
    f.close()