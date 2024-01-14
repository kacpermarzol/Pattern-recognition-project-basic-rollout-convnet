import timm
import torch
import torch.nn as nn
from torchsummary import summary
import model_vit
import resnet

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def create_teacher():
    model_teacher = model_vit.__dict__['vit_base_patch16'](
        num_classes=1000,
        drop_path_rate=0,
        global_pool=True,
    )

    checkpoint = torch.load('mae_finetuned_vit_base.pth', map_location='cpu')
    msg = model_teacher.load_state_dict(checkpoint['model'], strict = False)
    print(msg)
    model_teacher.eval()

    for param in model_teacher.parameters():
        param.requires_grad = False

    for block in model_teacher.blocks:
        block.attn.fused_attn = False
    return model_teacher


def create_student():
    model_student = resnet.ResNet(input_shape=[1, 3, 90, 90], depth=26, base_channels=6)  ## ~ 160k parameters
    # summary(model_student, (3, 90, 90))

    return model_student