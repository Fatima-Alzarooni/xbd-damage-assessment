# unet_model.py
# Siamese U-Net for building damage assessment
# takes pre and post disaster images as input
# outputs a segmentation mask with 4 damage classes

import torch
import torch.nn as nn


# conv block: two conv layers with batch norm and relu                
# this is the basic building block of the U-Net                      
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.block = nn.Sequential(
            # first conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            # second conv layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)
    


# encoder: 4 conv blocks with max pooling to downsample              
# shared between pre and post images                                  
# returns feature maps at each scale for skip connections            
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # each block doubles the channels and halves the spatial size
        self.block1 = ConvBlock(3,   32)   # 512x512 
        self.block2 = ConvBlock(32,  64)  # 256x256 
        self.block3 = ConvBlock(64, 128)  # 128x128 
        self.block4 = ConvBlock(128, 256)  # 64x64   
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # save feature maps at each scale for skip connections
        f1 = self.block1(x)           # 512x512, 64 channels
        f2 = self.block2(self.pool(f1))  # 256x256, 128 channels
        f3 = self.block3(self.pool(f2))  # 128x128, 256 channels
        f4 = self.block4(self.pool(f3))  # 64x64,   512 channels
        
        return f1, f2, f3, f4


# bottleneck: deepest part of the U-Net                              
# takes the most downsampled features and extracts                   
class Bottleneck(nn.Module):
    def __init__(self):
        super(Bottleneck, self).__init__()
        
        self.block = ConvBlock(256, 512)  # 32x32, 1024 channels
    
    def forward(self, x):
        return self.block(x)


# decoder: 4 upsampling blocks with skip connections                 
# takes difference features from encoder and bottleneck              
# upsamples back to 512x512                            
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # upsample using transposed convolution
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(128, 64,  kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(64,  32,  kernel_size=2, stride=2)
        
        # after each upsample concatenate the difference skip connection
        # in_channels is doubled
        self.conv1 = ConvBlock(512, 256)
        self.conv2 = ConvBlock(256, 128)
        self.conv3 = ConvBlock(128, 64)
        self.conv4 = ConvBlock(64,  32)
    
    def forward(self, bottleneck, diff1, diff2, diff3, diff4):
        # diff1-4 are the subtracted feature maps from the encoder
        
        x = self.up1(bottleneck)          # 32x32   -> 64x64
        x = self.conv1(torch.cat([x, diff4], dim=1))  # concat skip connection
        
        x = self.up2(x)                   # 64x64   -> 128x128
        x = self.conv2(torch.cat([x, diff3], dim=1))
        
        x = self.up3(x)                   # 128x128 -> 256x256
        x = self.conv3(torch.cat([x, diff2], dim=1))
        
        x = self.up4(x)                   # 256x256 -> 512x512
        x = self.conv4(torch.cat([x, diff1], dim=1))
        
        return x


# siamese u-net                              
# shared encoder for pre and post images                             
# subtracts feature maps at each scale                               
# decoder produces final segmentation mask                           

class SiameseUNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SiameseUNet, self).__init__()
        
        # shared encoder, same weights for both pre and post
        self.encoder    = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder    = Decoder()
        
        # final 1x1 conv to produce class scores
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
    
    def forward(self, pre, post):
        # encode both images with shared encoder
        pre_f1,  pre_f2,  pre_f3,  pre_f4  = self.encoder(pre)
        post_f1, post_f2, post_f3, post_f4 = self.encoder(post)
        
        # bottleneck on both
        pre_bottleneck  = self.bottleneck(self.encoder.pool(pre_f4))
        post_bottleneck = self.bottleneck(self.encoder.pool(post_f4))
        
        # subtract feature maps at each scale to get difference
        diff1 = torch.abs(pre_f1 - post_f1)
        diff2 = torch.abs(pre_f2 - post_f2)
        diff3 = torch.abs(pre_f3 - post_f3)
        diff4 = torch.abs(pre_f4 - post_f4)
        
        # combine bottleneck features
        bottleneck = torch.abs(pre_bottleneck - post_bottleneck)
        
        # decode
        x = self.decoder(bottleneck, diff1, diff2, diff3, diff4)
        
        # final classification
        return self.final_conv(x)


if __name__ == '__main__':
    # test
    model = SiameseUNet(num_classes=4)
    
    pre  = torch.randn(2, 3, 512, 512)
    post = torch.randn(2, 3, 512, 512)
    
    output = model(pre, post)
    print("Input shape:  ", pre.shape)
    print("Output shape: ", output.shape)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")