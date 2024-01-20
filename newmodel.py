import torch.nn as nn
from torchvision import transforms
from vit_rollout import VITAttentionRollout
import numpy as np
import torch
class NewModel(nn.Module):
    def __init__(self, teacher, student):
        super(NewModel, self).__init__()
        self.student = student
        self.teacher = teacher
        self.attention_rollout = VITAttentionRollout(teacher, head_fusion="max",
                                                discard_ratio=0.95)
    def forward(self, x):
        input_student = transforms.functional.resize(x, (90,90), antialias=True)
        # input_student = transforms.functional.resize(input_student, (224,224), antialias=True)

        target = self.attention_rollout(x)
        output = self.student(input_student)

        # return output, target
        return output, target
