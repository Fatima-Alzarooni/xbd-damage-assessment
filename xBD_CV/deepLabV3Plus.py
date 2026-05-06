# deepLabV3Plus.py
# siamese architecture using DeepLabV3+ for building damage assessment
# takes pre and post disaster images as input
# uses pre-trained ResNet50 encoder from smp
# outputs a segmentation mask with 4 damage classes

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# siamese deeplabv3+                             
# shared pre-trained encoder for pre and post images                 
# subtracts feature maps at each scale                               
# ASPP decoder produces final segmentation mask                      

class SiameseDeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=4):
        super(SiameseDeepLabV3Plus, self).__init__()
        
        # we instantiate the full smp model to get its pre-trained encoder,
        # its ASPP decoder, and its segmentation head.
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",      
            encoder_weights="imagenet",   # load the pre-trained weights
            in_channels=3,
            classes=num_classes
        )
        
    def forward(self, pre, post):
        # pass both images through the shared pre-trained encoder
        # smp encoders return a list of feature maps at different scales
        pre_features = self.model.encoder(pre)
        post_features = self.model.encoder(post)
        
        # subtract feature maps at each scale to get difference
        diff_features = []
        for f_pre, f_post in zip(pre_features, post_features):
            diff_features.append(torch.abs(f_pre - f_post))
            
        # pass the combined difference features to the ASPP decoder
        decoder_output = self.model.decoder(diff_features)
        
        # final 1x1 conv classification head
        masks = self.model.segmentation_head(decoder_output)
        
        return masks


if __name__ == '__main__':
    # test
    model = SiameseDeepLabV3Plus(num_classes=4)
    
    pre  = torch.randn(2, 3, 512, 512)
    post = torch.randn(2, 3, 512, 512)
    
    output = model(pre, post)
    print("Input shape:  ", pre.shape)
    print("Output shape: ", output.shape)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")