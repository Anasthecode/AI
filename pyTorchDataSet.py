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
print(dataset[100])


dataDir = "data/train"
targetClass = {v: k for k, v in ImageFolder(dataDir).class_to_idx.items()}


transform = transforms.Compose([
  transforms.Resize((128, 128)),
  transforms.ToTensor(),
])

dataset = PlayingCardDataset(dataDir, transform)
print(dataset[100])



class SimpleCardClassifier(nn.Module):
  def __init__(self, num_classes=53):
    super(SimpleCardClassifier, self).__init__()
    self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
    self.features = nn.Sequential(*list(self.base_model.children())[:-1])

    ennet_out_size = 1280
    self.classifier = nn.Linear(ennet_out_size, num_classes)
    self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ennet_out_size, num_classes)
        )

  def forward(self, x):
    x = self.features(x)
    output = self.classifier(x)
    return output

dataLoader = DataLoader(dataset, batch_size=32, shuffle=True)
model = SimpleCardClassifier()
for images, labels in dataLoader:
  break

exampleOutput = model(images)

print(exampleOutput.shape) # batchsize + num classes
#loss function
criterion = nn.CrossEntropyLoss()
#optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

criterion(exampleOutput, labels)

transform = transforms.Compose([
  transforms.Resize((128, 128)),
  transforms.ToTensor(),
])


trainFolder = "data/train"
validFolder = "data/valid"
testFolder = "data/test"


trainData = PlayingCardDataset(trainFolder, transform=transform)
validData = PlayingCardDataset(validFolder, transform=transform)
testData = PlayingCardDataset(testFolder, transform=transform)

trainLoader = DataLoader(trainData, batch_size=32, shuffle=True)
validLoader = DataLoader(validData, batch_size=32, shuffle=False)
TestLoader = DataLoader(testData, batch_size=32, shuffle=False)

numEpoch = 5
trainLosses, valLosses = [], []

model = SimpleCardClassifier(num_classes=53)

for epoch in range(numEpoch):

  model.train()
  runningLoss = 0.0
  for images, labels in trainLoader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    runningLoss += loss.item() + labels.size(0)
  
  trainLoss = runningLoss / len(trainLoader.dataset)
  trainLosses.append(trainLoss)

  model.eval()
  runningLoss = 0.0
  with torch.no_grad():
    for images, labels, in validLoader:
      outputs = model(images)
      loss = criterion(outputs, labels)
      runningLoss = loss.item() + labels.size(0)
    valLoss = runningLoss / len(validLoader.dataset)
    valLosses.append(valLoss)
    print(f"Epoch {epoch + 1}/{numEpoch} - Train loss: {trainLoss}, Valid loss: {valLoss}")

plt.plot(trainLosses, label='Training loss')
plt.plot(valLosses, label='Validation loss')
plt.legend()
plt.title("Loss over epochs")
plt.show()