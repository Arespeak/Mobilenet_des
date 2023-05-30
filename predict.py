import os
import torch
from torch import nn
import torchvision
from torchvision import models, transforms, datasets
import PIL.Image
from IPython.display import Image, display
import random

from img_process import process_img


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



def predict_img(path):

    model = models.mobilenet_v3_small()
    num_ftrs = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(num_ftrs, 10)  # No. of classes = 10

    model.load_state_dict(torch.load("checkpoint\mobilenet.pth", map_location="cuda:0"))

    model.eval()
    model.cuda()

    list_img_test = [img for img in os.listdir(path) if not img.startswith(".")]
    list_img_test.sort()

    transform = transforms.Compose([transforms.Resize((400, 400)),
                                    transforms.RandomRotation(10),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    file = random.choice(list_img_test)
    im_path = os.path.join(path_test, file)
    with PIL.Image.open(im_path) as im:
        im.show()
        im = transform(im)
        im = im.unsqueeze(0)
        output = model(im.cuda())
        proba = nn.Softmax(dim=1)(output)
        proba = [round(float(elem), 4) for elem in proba[0]]
        print(proba)
        print("Predicted class:", class_dict[proba.index(max(proba))])
        print("Confidence:", max(proba))
        proba2 = proba.copy()
        proba2[proba2.index(max(proba2))] = 0.
        print("2nd answer:", class_dict[proba2.index(max(proba2))])
        print("Confidence:", max(proba2))



path_test = "dataset/imgs/test"

predict_img(path_test)