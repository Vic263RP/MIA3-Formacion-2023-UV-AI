# Author: Antonio Martínez González
import torch
from torch import nn
from torch.nn import functional as F

class stack2convs(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(stack2convs, self).__init__()
        self.batch_norm = batch_norm
        self.conv2d_a = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=~self.batch_norm)
        self.conv2d_b = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=~self.batch_norm)
        if self.batch_norm:
            self.batchnorm_a = nn.BatchNorm2d(num_features=out_channels)
            self.batchnorm_b = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.conv2d_a(x)
        if self.batch_norm:
            x = self.batchnorm_a(x)
        x = F.relu(x)
        x = self.conv2d_b(x)
        if self.batch_norm:
            x = self.batchnorm_b(x)
        x = F.relu(x)
        return x

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feats_per_block=[64, 128, 256, 512], batch_norm=True, crop=True, logits=True):
        super(UNET, self).__init__()

        self.n_blocks = len(feats_per_block)
        self.crop = crop
        self.logits = logits
        self.contraction_path = nn.ModuleList()
        self.expansion_path_convs = nn.ModuleList()
        self.expansion_path_upsamplers = nn.ModuleList()

        # Build contraction path trainable layers
        for feats in feats_per_block:
            self.contraction_path.append(stack2convs(in_channels=in_channels, out_channels=feats, batch_norm=batch_norm))
            in_channels = feats

        # Build expansion path trainable layers
        for feats in feats_per_block[::-1]:
            self.expansion_path_convs.append(stack2convs(in_channels=feats*2, out_channels=feats, batch_norm=batch_norm))
            self.expansion_path_upsamplers.append(nn.ConvTranspose2d(in_channels=feats*2, out_channels=feats, kernel_size=2, stride=2))

        # Stack of 2D convs as bottleneck
        self.bottleneck = stack2convs(feats_per_block[-1], feats_per_block[-1]*2, batch_norm=batch_norm)
        # Point-wise 2D convolution as final model layer
        self.pointwise_conv = nn.Conv2d(feats_per_block[0], out_channels, kernel_size=1) 

    def forward(self, x):
        # Build an empty list to store the skip connections
        skips = []
        # Contraction path (encoder)
        for block in self.contraction_path:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Bottleneck
        x = self.bottleneck(x)
        # Expansion path (decoder)
        for i in range(self.n_blocks):
            x = self.expansion_path_upsamplers[i](x)
            skip = skips[-(i+1)]
            valid_height = x.size(2)
            valid_width = x.size(3)
            height_diff = skip.size(2) - valid_height
            width_diff = skip.size(3) - valid_width
            if height_diff or width_diff:
                if self.crop:
                    to_concat = skip[:,:,height_diff:height_diff+valid_height,width_diff:width_diff+valid_width]
                    x = torch.cat((to_concat, x), dim=1)
                else:
                    pad_left = width_diff // 2
                    pad_right = width_diff - pad_left
                    pad_top = height_diff // 2
                    pad_bottom = height_diff - pad_top
                    to_concat = F.pad(x, pad=(pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
                    x = torch.cat((skip, to_concat), dim=1)
            else:
                x = torch.cat((skip, x), dim=1)
            x = self.expansion_path_convs[i](x)
        # Final layer
        x = self.pointwise_conv(x)
        if not self.logits:
            x = F.sigmoid(x)
        return x