import os
import torch
from torch import nn
from torchvision import models, transforms, datasets
import PIL.Image
import random
import cv2

from predict import RealTimeProcess



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

def load_model():

    model = models.mobilenet_v3_small()
    num_ftrs = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(num_ftrs, 10)  # No. of classes = 10

    model.load_state_dict(torch.load("checkpoint\mobilenet.pth", map_location="cuda:0"))

    model.eval()
    model.cuda()

    return model

model = load_model()

camera = cv2.VideoCapture(0)

while True:

    ret, img = camera.read()

    cv2.imshow('carema', img)

    res = RealTimeProcess(img, model)
    print(res)
    print("Predicted class:", class_dict[res.index(max(res))])
    print("Confidence:", max(res))

    if cv2.waitKey(5) is 27:  # 按键盘的ESC键可退出，同时停顿一微秒
        break
    if cv2.getWindowProperty('camera', cv2.WND_PROP_AUTOSIZE) < 1:  # 用鼠标点击窗口退出键实现退出循环
        break


camera.release()

cv2.destroyAllWindows()
