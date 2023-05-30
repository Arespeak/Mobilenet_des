import os
import torch
from torch import nn
import torchvision
from torchvision import models, transforms, datasets


# device = torch.device("cuda:0")
#
# model_ft = models.mobilenet_v3_small(pretrained=True)
# model_ft = model_ft.to(device)

# print(model_ft)

model = models.mobilenet_v3_small()
num_ftrs = model.classifier[3].in_features

model.classifier[3] = nn.Linear(num_ftrs, 10)  # No. of classes = 10

model.load_state_dict(torch.load("checkpoint\mobilenet.pth", map_location="cuda:0"))

print(model)