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
        self.attention_rollout = VITAttentionRollout(teacher, head_fusion="max",
                                                discard_ratio=0.95)

    def forward(self, x):
        # input_student = transforms.functional.resize(x, (70,70), antialias=True)
        # input_student = transforms.functional.resize(input_student, (224,224), antialias=True)
        # attention_rollout = VITAttentionRollout(self.teacher, head_fusion="max",
        #                     discard_ratio=0.95)
        target = self.attention_rollout(x)

        output = self.student(x)

        # return output, target
        return output, target
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_false', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args
# def show_mask_on_image(img, mask):
#     img = np.float32(img) / 255
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     return np.uint8(255 * cam)

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    args = get_args()

    #This is the teacher model:
    # model_teacher = timm.create_model('deit3_large_patch16_224.fb_in1k', pretrained=True)
    model_teacher = torch.hub.load('facebookresearch/deit:main',
        'deit_tiny_patch16_224', pretrained=True)

    for param in model_teacher.parameters():
        param.requires_grad = False

    for block in model_teacher.blocks:
        block.attn.fused_attn = False

    model_teacher.to(device)
    model_teacher.eval()

    # model_student = resnet.ResNet(input_shape = [1,3,224,224], depth=26, base_channels=6) ## ~ 160k parameters
    model_student = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    num_ftrs = model_student.fc.in_features
    model_student.fc = nn.Linear(num_ftrs, 300)

    additional_layers = nn.Sequential(
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

    transform= transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    # data_folder = './data/ILSVRC2012_img_val'
    # training_data = ImageNet(data_folder, transform)
    data_folder = '/shared/sets/datasets/vision/ImageNet'
    imagenet_data = torchvision.datasets.ImageNet(data_folder, split='val', transform=transform)
    train_dataloader = DataLoader(imagenet_data, batch_size=1, shuffle=True, generator=torch.Generator(device=device))

    optimizer = torch.optim.Adam(model_student.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss().to(device)


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
            if (i+1) % 10000 == 0:
                print(f"STEP: {i}, loss: {loss.item()}")
                fig, axes = plt.subplots(1, 3, figsize=(10, 5))
                axes[0].imshow(image[0].permute(1, 2, 0).cpu().detach())
                axes[1].imshow(target.cpu())
                axes[2].imshow(output.cpu().detach())
                plt.savefig(f"train{epoch}_{i}.png")
                plt.close(fig)
        torch.save(model_student.state_dict(), 'model_state.pth')




