import timm
import torch
import torch.nn as nn

def create_teacher():
    #This is the teacher model:
    model_teacher = timm.create_model('deit3_large_patch16_224.fb_in1k', pretrained=True) # 300 million parameters
    # model_teacher = torch.hub.load('facebookresearch/deit:main',
    #     'deit_tiny_patch16_224', pretrained=True)

    for param in model_teacher.parameters():
        param.requires_grad = False

    for block in model_teacher.blocks:
        block.attn.fused_attn = False
    model_teacher.eval()
    return model_teacher


def create_student():
    # This is the student model:
    model_student = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)  # 11 million parameters
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

    return model_student