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

class ImageNet(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.image_list = os.listdir(data_folder)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.image_list[idx])
        image = Image.open(img_name)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

class NewModel(nn.Module):
    def __init__(self, teacher, student):
        super(NewModel, self).__init__()
        self.student = student
        self.teacher = teacher
        self.attention_rollout = VITAttentionRollout(teacher, head_fusion="mean",
                                                discard_ratio=0.95)
    def forward(self, x):
        input_student = transforms.functional.resize(x, (70,70), antialias=True)
        input_student = transforms.functional.resize(input_student, (224,224), antialias=True)

        target = self.attention_rollout(x)
        output = self.student(input_student)

        # return output, target
        return output, target


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
    print(device)

    #This is the teacher model:
    model_teacher = timm.create_model('deit3_large_patch16_224.fb_in1k', pretrained=True) # 300 million parameters
    # model_teacher = torch.hub.load('facebookresearch/deit:main',
    #     'deit_tiny_patch16_224', pretrained=True)

    for param in model_teacher.parameters():
        param.requires_grad = False

    for block in model_teacher.blocks:
        block.attn.fused_attn = False

    model_teacher.to(device)
    model_teacher.eval()

    #This is the studnet model:
    model_student = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) # 11 million parameters
    num_ftrs = model_student.fc.in_features
    model_student.fc = nn.Linear(num_ftrs, 400)

    additional_layers = nn.Sequential(
        nn.ReLU(),
        nn.Linear(400, 300),
        nn.ReLU(),
        nn.Linear(300, 196),
        nn.Sigmoid()
    )

    model_student = nn.Sequential(
        model_student,
        additional_layers
    )

    model_student = model_student.to(device)

    # architecture for training
    model = NewModel(model_teacher, model_student)
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # data_folder = './data/ILSVRC2012_img_val'
    # imagenet_data = ImageNet(data_folder, transform)
    data_folder = '/shared/sets/datasets/vision/ImageNet'
    imagenet_data = torchvision.datasets.ImageNet(data_folder, split='val', transform=transform)
    train_dataloader = DataLoader(imagenet_data, batch_size=1, shuffle=True, generator=torch.Generator(device=device))

    optimizer = torch.optim.Adam(model_student.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss().to(device)


    losses= []
    steps = []

    for epoch in range(10):
        print("EPOCH: ", epoch+1)
        for i, (image, _ ) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            image = image.to(device)
            optimizer.zero_grad()
            output, target = model(image)
            output = output.reshape(14,14)
            loss = criterion(target, output)
            loss.backward()
            optimizer.step()
            if (i+1) % 500 == 0:
                losses.append(loss.item())
                steps.append(epoch * 50000 + i+1)
            if (i+1) % 10000 == 0:
                print(f"STEP: {i+1}, loss: {loss.item()}")
                fig, axes = plt.subplots(1, 3, figsize=(10, 5))
                axes[0].imshow(image[0].permute(1, 2, 0).cpu().detach())
                axes[1].imshow(target.cpu())
                axes[2].imshow(output.cpu().detach())
                plt.savefig(f"train{epoch}_{i}.png")
                plt.close(fig)

        plt.plot(steps, losses)
        plt.title(f'Loss over time (after {epoch+1} epoch)')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig(f'Loss{epoch+1}.png')
        plt.close()
        torch.save(model_student.state_dict(), 'model_state.pth')


    # plt.plot(steps, losses)
    # plt.title('Loss over time')
    # plt.xlabel('Step')
    # plt.ylabel('Loss')
    # plt.savefig("Loss.png")
    # plt.close()


