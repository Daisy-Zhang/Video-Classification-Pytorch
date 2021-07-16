# Pytorch Video Classification
General video classification framework implemented by Pytorch for all video classification task.

(Remember first to extract all frames of your videos and put the frames in the same video data dir.)

## structure

### /checkpoints

This directory will store all models you trained.

### /data

Please put all your training or test data in this directory and follow the original directory structure.

### /log

All log record file will be stored in this directory.

### /models

You can put all the network model you design in this directory. I already provide four classic networks: **ALSTM**, **CNNLSTM**,**LRCN**, **RNN**.

### /simple_test

A simple test dataset created by me to validate the test function. Users can keep this directory and make their own change to use their test data, or just delete this directory.

This directory contains the original videos, their frames and the extracted feature.

## env

You could use following command to install all dependencies:

```python
pip -r requirements.txt
```

PS: for the **pytorch** version, early version may still be available.

## data

I implement Dataset class in **dataset.py**. All **videos and their frames** should put in **/data/test** and **/data/train** directory.  The child directory is each classes, such as **/data/test/ClassA** et al.

## config

Users can change the config setting in **conf.py** as they need, such as IMAGE_SIZE, EPOCH et al.

## feature extraction

After you put your video and their frames well in **/data**, users should first run **extract_features.py** to extract the feature of your video data:

```python
>> python extract_features.py -h
usage: extract_features.py [-h] -model MODEL -seq_length SEQ_LENGTH
                           -image_size IMAGE_SIZE -sample_frames SAMPLE_FRAMES
                           -data_dir DATA_DIR [-gpu]

optional arguments:
  -h, --help            show this help message and exit
  -model MODEL          extractor model type
  -seq_length SEQ_LENGTH
                        sample frames length
  -image_size IMAGE_SIZE
                        crop image size
  -sample_frames SAMPLE_FRAMES
                        sampled frames
  -data_dir DATA_DIR    whole image folder dir
  -gpu                  use gpu or not
```

Once feature extraction done, there will be a **/sequences** directory in **/data** which contains all the features and the meta files.

## train

Users can run **train.py** to start training:

```python
>> python train.py -h
usage: train.py [-h] -model MODEL -seq_dir SEQ_DIR -seq_length SEQ_LENGTH
                -cnn_type CNN_TYPE [-gpu]

optional arguments:
  -h, --help            show this help message and exit
  -model MODEL          model type
  -seq_dir SEQ_DIR      features dir
  -seq_length SEQ_LENGTH
                        sequences length
  -cnn_type CNN_TYPE    features extractor cnn type
  -gpu                  use gpu or not
```

## test

Users can run **test.py** to evaluate your model:

```python
>> python test.py -h
usage: test.py [-h] -model MODEL -weights WEIGHTS [-gpu] -data_path DATA_PATH
               -image_size IMAGE_SIZE -cnn_type CNN_TYPE -seq_length
               SEQ_LENGTH

optional arguments:
  -h, --help            show this help message and exit
  -model MODEL          model type
  -weights WEIGHTS      the weights file you want to test
  -gpu                  use gpu or not
  -data_path DATA_PATH  test data path
  -image_size IMAGE_SIZE
                        input image size
  -cnn_type CNN_TYPE    lstm feature extractor type
  -seq_length SEQ_LENGTH
                        sequences length
```



## others

**utils.py**: some utils function used in train.py and test.py. Users can modify this file for their convinence.



If this repo do you a favor, a star is my pleasure :)

And if you find any problem, please contact me or open an issue.