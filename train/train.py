#! /usr/bin/env python3

"""
Train object detector (YoloV5)
"""

import torch
from torchvision import transforms
from torchvision.models import yolov5
from torch.utils.data import DataLoader

from torch.optim import Adam
from torch.nn import BCELoss
from PIL import Image as PILImage

from train.CoralSpawnDataset import CoralSpawnDataset

# load YOLOv5 model
model = yolov5.yolov5m()

# load the weights
weights_file = 'weights/yolov5m-seg.pt'
weights = torch.load(weights_file)
model.load_state_dict(weights)

# move model to GPU
model = model.to('cuda') # assumes cuda is available

# freeze all layers, except the last layer:
for name, param in model.parameters():
    if 'last_' not in name:
        param.requiresGrad = False
        
# define optimizer and loss function
optimizer = Adam(filter(lambda p: p.requiresGrad, model.parameters()), lr = 0.001)
loss_fn = BCELoss()

# load in the training data:
train_dataset = CoralSpawnDataset()
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# training loop
num_epochs = 5
for epoch in range(num_epochs):
    for i, target in enumerate(train_dataloader):
        image, sample = target
        image = image.to('cuda')
        labels = labels.to('cuda')
        poly = poly.to('cuda')
        
        # forward pass
        outputs = model(image, sample)
        
        # compute loss
        loss = loss_fn(outputs, labels)
        
        # backward pass and optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

