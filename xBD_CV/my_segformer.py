# SegFormer for building damage assessment
# takes 6-channel input (pre + post concatenated)
# uses pretrained SegFormer backbone from segmentation-models-pytorch
# output: 4-class damage segmentation mask

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

class SegFormer(nn.Module):
    def __init__(self, num_classes=4, pretrained="nvidia/mit-b2"):
        super().__init__()

        # load pretrained segformer
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            pretrained,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

        # replace first conv layer to accept 6 channels instead of 3
        # get the original first conv
        old_conv = self.segformer.segformer.encoder.patch_embeddings[0].proj

        # create new conv with 6 input channels, same everything else
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding
        )

        # initialize new conv: copy pretrained weights for first 3 channels
        # and repeat them for the next 3 (pre and post share same visual domain)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            new_conv.weight[:, 3:, :, :] = old_conv.weight
            new_conv.bias = old_conv.bias

        # replace the layer
        self.segformer.segformer.encoder.patch_embeddings[0].proj = new_conv

    def forward(self, pre, post):
        # concatenate pre and post along channel dim → (batch, 6, H, W)
        x = torch.cat([pre, post], dim=1)

        outputs = self.segformer(pixel_values=x)

        # segformer outputs at H/4, W/4 so upsample back to full size
        logits = outputs.logits  # (batch, num_classes, H/4, W/4)
        logits = F.interpolate(
            logits,
            size=pre.shape[-2:],  # (H, W)
            mode='bilinear',
            align_corners=False
        )

        return logits


if __name__ == '__main__':
    model = SegFormer(num_classes=4)
    pre   = torch.randn(2, 3, 256, 256)
    post  = torch.randn(2, 3, 256, 256)
    out   = model(pre, post)
    print("Output shape:", out.shape)  # should be (2, 4, 256, 256)