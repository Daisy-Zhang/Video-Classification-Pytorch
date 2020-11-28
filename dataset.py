import os
import sys

import numpy 
import torch
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import conf


def getDataLoader(seq_dir, info_list, seq_length, cnn_type, transform = None):
    my_dataset = MyDataset(info_list, seq_dir, seq_length, cnn_type, transform)

    #my_dataset.test()

    my_dataloader = torch.utils.data.DataLoader(
        my_dataset,
        batch_size = conf.TRAINING_BATCH_SIZE,
        shuffle = True,
        num_workers = conf.NUM_WORKERS
    )

    return my_dataloader

class MyDataset(Dataset):
    def __init__(self, info_list, seq_dir, seq_length, cnn_type, transform = None):
        """
        Args:
            info_list (string): Path to the info list file with annotations.
            seq_dir (string): Directory with all the extracted features.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(info_list, delimiter = ' ', header = None)
        self.seq_dir = seq_dir
        self.transform = transform
        self.seq_length = seq_length
        self.cnn_type = cnn_type
            
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        video_name = self.landmarks_frame.iloc[idx, 0]
        seq_path = os.path.join(self.seq_dir, video_name[0 : len(video_name) - 4] + '-' + str(self.seq_length) + '-features-' + str(self.cnn_type) + '.npy')

        label = self.landmarks_frame.iloc[idx,1]
        
        features = numpy.load(seq_path)

        if self.transform:
            features = self.transform(features)
        return features, label
    
    def test(self):
        print(len(self.landmarks_frame))
        print(self.landmarks_frame)
        print(type(self.landmarks_frame))
        print(self.landmarks_frame.iloc[0, 0])
        print(self.landmarks_frame.iloc[1, 0])