import torch
import torch.nn as nn # specific functions
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision # Useful for images
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm # Also for useful for images

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from tqdm.notebook import tqdm

class PlayingCardDataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data = ImageFolder(data_dir, transform= transform)

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    return self.data[index]
  

  @property
  def classes(self):
    return self.data.classes
  
dataset = PlayingCardDataset(data_dir="/home/abdulaal/AI/data/train")



print(len(dataset))