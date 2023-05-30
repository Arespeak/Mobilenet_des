import os
import shutil
import time
from tqdm import tqdm
import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# import pandas as pd
import PIL.Image
from IPython.display import Image
# from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, datasets

path_train = "dataset/imgs/train"
path_test = "dataset/imgs/test"

class_dict = {0 : "safe driving",
              1 : "texting - right",
              2 : "talking on the phone - right",
              3 : "texting - left",
              4 : "talking on the phone - left",
              5 : "operating the radio",
              6 : "drinking",
              7 : "reaching behind",
              8 : "hair and makeup",
              9 : "talking to passenger"}

def process_img(path, is_train=True):

    transform = transforms.Compose([transforms.Resize((400, 400)),
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if is_train:
        classes = [c for c in os.listdir(path) if not c.startswith(".")]
        classes.sort()
        print(classes)

        data = datasets.ImageFolder(root=path, transform=transform)
        total_len = len(data)
        training_len = int(0.8 * total_len)
        valid_len = total_len - training_len

        training_data, valid_data = torch.utils.data.random_split(data, (training_len, valid_len))

        train_loader = torch.utils.data.DataLoader(dataset=training_data,
                                                   batch_size=32,
                                                   shuffle=True,
                                                   drop_last=False,
                                                   num_workers=2)

        valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                                  batch_size=32,
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=2)
        return train_loader, valid_loader

    # else:


