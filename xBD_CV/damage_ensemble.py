# ensemble.py
# soft voting ensemble combining Siamese U-Net, Siamese DeepLabV3+, and SegFormer
# averages softmax probabilities from all 3 models before argmax
# no training needed, just load checkpoints and run inference

import torch
import torch.nn.functional as F
from torch.amp import autocast

from unet_model import SiameseUNet
from deepLabV3Plus import SiameseDeepLabV3Plus
from my_segformer import SegFormer


# load all 3 models from their checkpoints
def load_ensemble(unet_ckpt, deeplab_ckpt, segformer_ckpt, num_classes=4, device='cuda'):

    print("Loading Siamese U-Net...")
    unet = SiameseUNet(num_classes=num_classes).to(device)
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device, weights_only=False)['model_state_dict'])
    unet.eval()

    print("Loading Siamese DeepLabV3+...")
    deeplab = SiameseDeepLabV3Plus(num_classes=num_classes).to(device)
    deeplab.load_state_dict(torch.load(deeplab_ckpt, map_location=device, weights_only=False)['model_state_dict'])
    deeplab.eval()

    print("Loading SegFormer...")
    segformer = SegFormer(num_classes=num_classes).to(device)
    segformer.load_state_dict(torch.load(segformer_ckpt, map_location=device, weights_only=False)['model_state_dict'])
    segformer.eval()

    print("All models loaded!")
    return unet, deeplab, segformer


# soft voting: average softmax probs from all 3 models
def ensemble_predict(unet, deeplab, segformer, pre, post):

    with torch.no_grad():
        with autocast('cuda'):
            out_unet     = F.softmax(unet(pre, post),     dim=1)
            out_deeplab  = F.softmax(deeplab(pre, post),  dim=1)
            out_segformer = F.softmax(segformer(pre, post), dim=1)

    # average probabilities across all 3 models
    avg_probs = (out_unet + out_deeplab + out_segformer) / 3.0

    return avg_probs  # (batch, num_classes, H, W)