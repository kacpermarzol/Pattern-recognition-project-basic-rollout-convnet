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
from torch.utils.data import DataLoader
import resnet
import imagenet
from utils import *
import newmodel


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
    print(device)

    model_teacher = create_teacher()
    model_teacher.to(device)

    model_student = create_student()
    model_student = model_student.to(device)

    weights_path = './evaluation/model_state.pth'
    # checkpoint = (torch.load(weights_path) if device != 'cpu' else torch.load(weights_path, map_location=torch.device('cpu')))
    checkpoint = torch.load(weights_path, map_location=device)
    model_student.load_state_dict(checkpoint)
    model = model_student.to(device)

    model = newmodel.NewModel(model_teacher, model_student)
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


    data_folder = './data/test'
    imagenet_data = imagenet.ImageNet(data_folder, transform)

    # data_folder = '/shared/sets/datasets/vision/ImageNet'
    # imagenet_data = torchvision.datasets.ImageNet(data_folder, split='val', transform=transform)

    train_dataloader = DataLoader(imagenet_data, batch_size=1, shuffle=False, generator=torch.Generator(device=device))


    topil = transforms.ToPILImage()
    with torch.no_grad():
        img = img.unsqueeze(0)
        img2 = img2.unsqueeze(0)

        output, target = model(img)
        output = output.reshape(14,14)
        output = topil(output)
        target = topil(target)
        output2, target2 = model(img2)
        output2 = output2.reshape(14, 14)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # axes[0][0].imshow(img[0].permute(1, 2, 0).cpu().detach())
        axes[0].imshow(target)
        axes[1].imshow(output)
        # axes[1][0].imshow(img2[0].permute(1, 2, 0).cpu().detach())
        # axes[1][1].imshow(target2)
        # axes[1][2].imshow(output2)
        plt.show()
        plt.close()

        for i, image in tqdm(enumerate(train_dataloader), total=10):
            image = image.to(device)
            output, target = model(image)
            output = output.reshape(14, 14)

            fig, axes = plt.subplots(1, 3, figsize=(10, 5))
            axes[0].imshow(image[0].permute(1, 2, 0).cpu().detach())
            axes[1].imshow(target.cpu())
            axes[2].imshow(output.cpu().detach())
            plt.show()
            plt.close(fig)


