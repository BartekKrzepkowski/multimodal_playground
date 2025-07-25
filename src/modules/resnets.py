import torch
import torch.nn as nn
import torchvision.models as models

from src.utils.utils_model import infer_dims_from_blocks

class BimodalResNet(nn.Module):
    def __init__(self, resnet_type, num_classes, img_height, img_width, fusion='concat', pretrained=False):
        super().__init__()
        # Stwórz dwa "głowy" ResNet18, do layer2 włącznie:
        resnet_constructor = getattr(models, resnet_type)
        base1 = resnet_constructor(pretrained=pretrained)
        base2 = resnet_constructor(pretrained=pretrained)
        # Głowy do layer2 (conv1, bn1, relu, maxpool, layer1, layer2)
        self.left_branch = nn.Sequential(
            base1.conv1, base1.bn1, base1.relu, base1.maxpool,
            base1.layer1, base1.layer2
        )
        self.right_branch = nn.Sequential(
            base2.conv1, base2.bn1, base2.relu, base2.maxpool,
            base2.layer1, base2.layer2
        )
        # Połączenie: output z left_branch i right_branch to shape [B, 128, H/4, W/4] (dla CIFAR/ImageNet rozmiarów)
        self.fusion = fusion

        # Wspólna dalsza część ResNet18:
        # Dla concatenacji liczba kanałów będzie 256, dla sum/mean/max zostaje 128
        if fusion == 'concat':
            in_channels = self.left_branch[-1][-1].conv2.out_channels * 2  # typowo 128*2=256
        else:
            in_channels = self.left_branch[-1][-1].conv2.out_channels

        z = torch.randn(1, 3, img_height, img_width)
        self.channels_out, self.height, self.width = infer_dims_from_blocks(self.left_branch, z)
        # Skopiuj layer3 i layer4 (odpowiednio zmodyfikuj input channels jeśli concat)
        # Musimy zmodyfikować pierwszy blok layer3, by miał wejście 256 kanałów (concat)
        base_rest = resnet_constructor(pretrained=pretrained)
        self.layer3 = self._make_layer3(base_rest, in_channels)
        self.layer4 = base_rest.layer4
        self.avgpool = base_rest.avgpool

        # Rozmiar wejścia do fc określamy inferując przez warstwy:
        dummy = torch.zeros(1, in_channels, self.height, self.width)
        was_training3 = self.layer3.training
        was_training4 = self.layer4.training
        self.layer3.eval()
        self.layer4.eval()
        with torch.no_grad():
            y = self.layer3(dummy)
            y = self.layer4(y)
            y = self.avgpool(y)
        self.layer3.train(was_training3)
        self.layer4.train(was_training4)
        fc_in_features = y.view(1, -1).shape[1]
        self.fc = nn.Linear(fc_in_features, num_classes)

    def _make_layer3(self, base_resnet, in_channels):
        layer3 = base_resnet.layer3
        first_block = list(layer3.children())[0]
        expected_in_channels = first_block.conv1.in_channels
        if in_channels != expected_in_channels:
            # Nowy conv1
            first_block.conv1 = nn.Conv2d(
                in_channels, first_block.conv1.out_channels,
                kernel_size=3, stride=first_block.conv1.stride,
                padding=first_block.conv1.padding, bias=False
            )
            # Nowy downsample
            if first_block.downsample is not None:
                first_block.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_channels, first_block.downsample[0].out_channels,
                        kernel_size=1, stride=first_block.downsample[0].stride, bias=False
                    ),
                    nn.BatchNorm2d(first_block.downsample[1].num_features)
                )
            blocks = [first_block] + list(layer3.children())[1:]
            return nn.Sequential(*blocks)
        return layer3

    def forward(self, x1, x2, enable_left_branch=True, enable_right_branch=True):
        if enable_left_branch:
            x1 = self.left_branch(x1)  # [B, 128, H, W]
        else:
            x1 = torch.zeros((x1.size(0), self.channels_out, self.height, self.width), device=x1.device)

        if enable_right_branch:
            x2 = self.right_branch(x2)  # [B, 128, H, W]
        else:
            x2 = torch.zeros((x2.size(0), self.channels_out, self.height, self.width), device=x2.device)

        # Fusion
        if self.fusion == 'concat':
            x = torch.cat([x1, x2], dim=1)  # [B, 256, H, W]
        elif self.fusion == 'sum':
            x = x1 + x2
        elif self.fusion == 'mean':
            x = (x1 + x2) / 2
        elif self.fusion == 'max':
            x = torch.max(x1, x2)
        else:
            raise ValueError(f'Unknown fusion type: {self.fusion}')
        # Dalej jak ResNet18
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out

