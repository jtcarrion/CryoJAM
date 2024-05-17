import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Define the encoder path (downsampling)
        self.down_conv1 = self.contract_block(2, 64, 3, 1)
        self.down_conv2 = self.contract_block(64, 128, 3, 1)
        self.down_conv3 = self.contract_block(128, 256, 3, 1)
        self.down_conv4 = self.contract_block(256, 512, 3, 1)

        # Define the decoder path (upsampling)
        self.up_conv4 = self.expand_block(512, 256, 3, 1)
        self.up_conv3 = self.expand_block(256*2, 128, 3, 1)
        self.up_conv2 = self.expand_block(128*2, 64, 3, 1)
        self.up_conv1 = self.expand_block(64*2, 64, 3, 1)

        # Final output layer
        self.final_conv = nn.Conv3d(64, 1, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(1)

        # Additional convolutional layer
        self.conv64 = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    
    def __call__(self, x):
        # Encoder pathway
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(F.max_pool3d(x1, kernel_size=2, stride=2))
        x3 = self.down_conv3(F.max_pool3d(x2, kernel_size=2, stride=2))
        x4 = self.down_conv4(F.max_pool3d(x3, kernel_size=2, stride=2))

        # Decoder pathway
        x = self.up_conv4(F.interpolate(x4, scale_factor=2))
        x = self.up_conv3(F.interpolate(torch.cat((x, x3), 1), scale_factor=2))
        x = self.up_conv2(F.interpolate(torch.cat((x, x2), 1), scale_factor=2))
        x = self.up_conv1(F.interpolate(torch.cat((x, x1), 1), scale_factor=2))

        # Final output
        x = self.final_conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.conv64(x)
        x = self.sigmoid(x)
        return x

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels)
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(out_channels)
        )
        return expand
