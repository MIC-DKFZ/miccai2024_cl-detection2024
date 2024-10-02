"""
Project: CL-Detection2024 Challenge Baseline
============================================

Model loading
模型载入

Email: zhanghongyuan2017@email.szu.edu.cn
"""

import torch
import torch.nn as nn
#from ultralytics import YOLO
from torchvision import models
#import monai
import segmentation_models_pytorch as smp
import timm
from torchvision.models import ResNet50_Weights

class LUConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(out_channels)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def make_n_conv_layer(in_channels, depth, double_channel=False):
    if double_channel:
        layer1 = LUConv(in_channels, 32 * (2 ** (depth + 1)))
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)))
    else:
        layer1 = LUConv(in_channels, 32 * (2**depth))
        layer2 = LUConv(32 * (2**depth), 32 * (2**depth) * 2)

    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channels, depth):
        super(DownTransition, self).__init__()
        self.ops = make_n_conv_layer(in_channels, depth)
        self.pool = nn.MaxPool2d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.pool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.ops = make_n_conv_layer(
            in_channels + out_channels // 2, depth, double_channel=True
        )

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, in_channels, n_labels):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv2d(in_channels, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_class=1):
        super(UNet, self).__init__()

        self.down_tr64 = DownTransition(in_channels, 0)
        self.down_tr128 = DownTransition(64, 1)
        self.down_tr256 = DownTransition(128, 2)
        self.down_tr512 = DownTransition(256, 3)

        self.up_tr256 = UpTransition(512, 512, 2)
        self.up_tr128 = UpTransition(256, 256, 1)
        self.up_tr64 = UpTransition(128, 128, 0)
        self.out_tr = OutputTransition(64, n_class)

    def forward(self, x):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128, self.skip_out128 = self.down_tr128(self.out64)
        self.out256, self.skip_out256 = self.down_tr256(self.out128)
        self.out512, self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512, self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)

        return self.out

class UNetResNet50(nn.Module):
    def __init__(self, n_classes=1, pretrained=True):
        super(UNetResNet50, self).__init__()
        
        # Load ResNet50 pre-trained on ImageNet1k
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,)
        
        # Encoder layers (use the layers from the pretrained ResNet-50)
        self.encoder1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.encoder2 = self.resnet.layer1
        self.encoder3 = self.resnet.layer2
        self.encoder4 = self.resnet.layer3
        self.encoder5 = self.resnet.layer4
        
        # Decoder layers
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(2048, 1024)
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(1024, 512)
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(512, 256)
        self.upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        # Final Convolution layer
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        """A Convolutional Block for the U-Net decoder"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)
        
        # Decoder path
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # Final Convolution
        out = self.final_conv(dec1)
        return out
    

def load_model(model_name, **kwargs):
    """
    This function loads the specified model based on the provided model name.
    该函数根据提供的模型名称加载指定的模型。
    Args:
        model_name (str): The name of the model to load. Currently, the only valid option is 'UNet'.
                          要加载的模型名称。目前唯一有效的选项是'UNet'。
    Returns:
        nn.Module: The corresponding model.
                   相应的模型。
    """
    N_CLASS = 53 # number of landmarks
    if model_name == "UNet_custom":
        model = UNet(in_channels=3, n_class=N_CLASS)
    elif model_name == "UNet_smp":
        model = smp.Unet(encoder_name=kwargs.get('encoder_name', 'resnet18'), encoder_weights=None, in_channels=1, classes=N_CLASS, activation=kwargs.get('activation', 'sigmoid'))
    elif model_name == "Deeplab":
        model = smp.DeepLabV3Plus(encoder_name=kwargs.get('encoder_name', 'resnet18'), encoder_weights=None, in_channels=1, classes=N_CLASS, activation=kwargs.get('activation', 'sigmoid'))
    elif model_name == "21k":
        model = UNetResNet50(n_classes=1, pretrained=True)
    elif model_name == "ResNet50_fcn":
        model = models.segmentation.fcn_resnet50(progress=True, num_classes=N_CLASS, **kwargs) # TODO: write wrapper so that output is a tensor
    elif model_name == "ResNet50_deeplabv3":
        model = models.segmentation.deeplabv3_resnet50(progress=True, num_classes=N_CLASS, **kwargs)

    else:
        raise ValueError(
            "Please input valid model name, {} not in model zones.".format(model_name)
        )
    return model


if __name__ == "__main__":
    model_names = [
                    #"UNet",
                    "UNet_smp"
                    # "ResNet50_fcn",
                    # "ResNet50_deeplabv3",
                    # "SegResNet18",
                    # "SegResNet34",
                    # "SegResNetDS18",
                    # "SegResNetDS50"
                    ]
    for model_name in model_names:
        if model_name == "UNet_smp":
            # encoder_names = ["resnet18", "resnet50", "mobilenet_v2"]
            kwargs = {"encoder_name": "resnet34"}
            model = load_model(model_name=model_name, **kwargs)
        else:
            model = load_model(model_name=model_name)
        print(model_name)

        x = torch.randn(2, 3, 512, 512)
        y = model(x)
        if isinstance(y, dict):
            print(y.keys())
            print(y["out"].shape)
        else:
            print(y.shape)
