import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import getModel, WarmUpLR
from dataset import getDataLoader
import conf

def train(model, epoch, train_loader, loss_function, optimizer, warmup_scheduler, use_gpu):
    model.train()
    for batch_index, (sequences, labels) in enumerate(train_loader):
        if epoch <= conf.WARM_EPOCH:
            warmup_scheduler.step()
        
        #print(sequences.size())
        #sequences = sequences.reshape(sequences.size()[0], sequences.size()[1], sequences.size()[2] * sequences.size()[3] * sequences.size()[4])

        sequences = Variable(sequences)
        labels = Variable(labels)

        if use_gpu:
            labels = labels.cuda()
            sequences = sequences.cuda()

        optimizer.zero_grad()
        #print(sequences.size())
        outputs = model(sequences)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * conf.TRAINING_BATCH_SIZE + len(sequences),
            total_samples=len(train_loader.dataset)
        ))

    return

def eval(model, epoch, val_loader, loss_function, use_gpu):
    model.eval()

    loss = 0.0
    correct = 0.0

    for (sequences, labels) in val_loader:
        #sequences = sequences.reshape(sequences.size()[0], sequences.size()[1], sequences.size()[2] * sequences.size()[3] * sequences.size()[4])

        sequences = Variable(sequences)
        labels = Variable(labels)

        if use_gpu:
            sequences = sequences.cuda()
            labels = labels.cuda()

        outputs = model(sequences)
        loss = loss_function(outputs, labels)
        loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        loss / len(val_loader.dataset),
        correct.float() / len(val_loader.dataset)
    ))
    
    return correct.float() / len(val_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type = str, required = True, help = 'model type')
    parser.add_argument('-seq_dir', type = str, required = True, help = 'features dir')
    parser.add_argument('-seq_length', type = int, required = True, help = 'sequences length')
    parser.add_argument('-cnn_type', type = str, required = True, help = 'features extractor cnn type')
    parser.add_argument('-gpu', action="store_true", help = 'use gpu or not')
    args = parser.parse_args()

    print(args.model)
    print(args.gpu)

    model = getModel(model_type = args.model, use_gpu = args.gpu)

    train_loader = getDataLoader(args.seq_dir, args.seq_dir + '/train_metadata.txt', args.seq_length, args.cnn_type)
    print('get train loader done')
    val_loader = getDataLoader(args.seq_dir, args.seq_dir + '/test_metadata.txt', args.seq_length, args.cnn_type)
    print('get val loader done')

    checkpoints_path = os.path.join(conf.CHECKPOINTS_PATH, args.model, datetime.now().isoformat())
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    checkpoints_path = os.path.join(checkpoints_path, '{model}-{epoch}-{type}.pth')

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=conf.LEARNING_RATE, momentum=conf.MOMENTUM, weight_decay=conf.WEIGHT_DECAY)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.MILESTONES, gamma=conf.GAMMA)
    iter_per_epoch = len(train_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * conf.WARM_EPOCH)

    best_acc = 0.0
    for epoch in range(1, conf.EPOCH):
        if epoch > conf.WARM_EPOCH:
            train_scheduler.step(epoch)
        
        train(model, epoch, train_loader, loss_function, optimizer, warmup_scheduler, args.gpu)
        acc = eval(model, epoch, val_loader, loss_function, args.gpu)

        if best_acc < acc:
            torch.save(model.state_dict(), checkpoints_path.format(model=args.model, epoch=epoch, type='best'))
            best_acc = acc
            continue

        #if not epoch % conf.SAVE_EPOCH:
        #    torch.save(model.state_dict(), checkpoints_path.format(model=args.model, epoch=epoch, type='regular'))