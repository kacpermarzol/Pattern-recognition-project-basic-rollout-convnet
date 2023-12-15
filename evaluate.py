import argparse
import sys
import torch
from PIL import Image
import torchvision
from torchvision import transforms
from torchsummary import summary
import numpy as np
# import cv2
import timm
import os
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
import matplotlib.pyplot as plt

from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout
from torch.utils.data import DataLoader
import resnet


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
    print(device)

    # model_student = resnet.ResNet(input_shape = [1,3,224,224], depth=26, base_channels=6) ## ~ 160k parameters
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 300)

    additional_layers = nn.Sequential(
        nn.ReLU(),
        nn.Linear(300, 196),
        nn.Sigmoid()
    )

    model = nn.Sequential(
        model,
        additional_layers
    )

    weights_path = './evaluation/model_state.pth'
    # checkpoint = (torch.load(weights_path) if device != 'cpu' else torch.load(weights_path, map_location=torch.device('cpu')))
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image_path = './examples/plane.png'
    img = Image.open(image_path)
    img2 = img.crop((0, 0, 200, 160))

    img = transform(img)
    img2 = transform(img2)

    img = img.to(device)
    img2 = img2.to(device)

    # plt.imshow(img.permute(1,2,0))
    # plt.show()
    #

    # data_folder = '/shared/sets/datasets/vision/ImageNet'
    # imagenet_data = torchvision.datasets.ImageNet(data_folder, split='val', transform=transform)
    # train_dataloader = DataLoader(imagenet_data, batch_size=1, shuffle=True, generator=torch.Generator(device=device))

    with torch.no_grad():
        img = img.unsqueeze(0)
        img2 = img2.unsqueeze(0)

        output = model(img)
        output = output.reshape(14,14)
        output2 = model(img2)
        output2 = output2.reshape(14, 14)

        fig, axes = plt.subplots(1, 4, figsize=(10, 5))

        axes[0].imshow(img[0].permute(1, 2, 0).cpu().detach())
        axes[1].imshow(output)
        axes[2].imshow(img2[0].permute(1, 2, 0).cpu().detach())
        axes[3].imshow(output2)

        plt.show()

