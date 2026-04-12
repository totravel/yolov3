import torch
from torch import nn
import timm
from utils import init_weights


class CBL(nn.Module):
  """Conv2d + BatchNorm2d + LeakyReLU"""
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    super().__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    self.bn = nn.BatchNorm2d(out_channels)
    self.lrelu = nn.LeakyReLU(0.1, inplace=True)

  def forward(self, x):
    x = self.lrelu(self.bn(self.conv(x)))
    return x


class BasicBlock(nn.Module):
  def __init__(self, in_channels, mid_channels, out_channels):
    super().__init__()
    self.cbl1 = CBL(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
    self.cbl2 = CBL(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
  
  def forward(self, x):
    return self.cbl2(self.cbl1(x)) + x


class Backbone(nn.Module):
  def __init__(self):
    super().__init__()
    self.cbl = CBL(3, 32, kernel_size=3, stride=1, padding=1)
    self.layer1 = self._make_layer(32, 64, 1)
    self.layer2 = self._make_layer(64, 128, 2)
    self.layer3 = self._make_layer(128, 256, 8)
    self.layer4 = self._make_layer(256, 512, 8)
    self.layer5 = self._make_layer(512, 1024, 4)

  def _make_layer(self, in_channels, out_channels, num_blocks):
    layers = [CBL(in_channels, out_channels, kernel_size=3, stride=2, padding=1)]  # Downsample
    for _ in range(num_blocks):
      layers.append(BasicBlock(out_channels, in_channels, out_channels))
    return nn.Sequential(*layers)
  
  def forward(self, x):  # (N, 3, 416, 416)
    x = self.cbl(x)      # (N, 32, 416, 416)
    x = self.layer1(x)   # (N, 64, 208, 208)
    x = self.layer2(x)   # (N, 128, 104, 104)
    y1 = self.layer3(x)  # (N, 256, 52, 52)
    y2 = self.layer4(y1) # (N, 512, 26, 26)
    y3 = self.layer5(y2) # (N, 1024, 13, 13)
    return y1, y2, y3


class Neck(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer3 = self._make_layer(1024, 1024, 512)

    self.cbl2 = CBL(512, 256, 1)
    self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
    self.layer2 = self._make_layer(768, 512, 256)

    self.cbl1 = CBL(256, 128, 1)
    self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
    self.layer1 = self._make_layer(384, 256, 128)
  
  def _make_layer(self, in_channels, mid_channels, out_channels):
    return nn.Sequential(
      CBL(in_channels, out_channels, kernel_size=1),  # reduction
      CBL(out_channels, mid_channels, kernel_size=3, padding=1),
      CBL(mid_channels, out_channels, kernel_size=1),  # reduction
      CBL(out_channels, mid_channels, kernel_size=3, padding=1),
      CBL(mid_channels, out_channels, kernel_size=1)  # reduction
    )
  
  def forward(self, y1, y2, y3):
    y3 = self.layer3(y3)

    y2 = torch.cat([self.up2(self.cbl2(y3)), y2], 1)
    y2 = self.layer2(y2)

    y1 = torch.cat([self.up1(self.cbl1(y2)), y1], 1)
    y1 = self.layer1(y1)
    return y1, y2, y3


class Head(nn.Module):
  def __init__(self, num_anchors, num_classes):
    super().__init__()
    out_channels = num_anchors * (5 + num_classes)
    self.layer1 = self._make_layer(128, 256, out_channels)
    self.layer2 = self._make_layer(256, 512, out_channels)
    self.layer3 = self._make_layer(512, 1024, out_channels)

  def _make_layer(self, in_channels, mid_channels, out_channels):
    return nn.Sequential(
      CBL(in_channels, mid_channels, kernel_size=3, padding=1),
      nn.Conv2d(mid_channels, out_channels, kernel_size=1)
    )
  
  def forward(self, y1, y2, y3):
    y1 = self.layer1(y1)
    y2 = self.layer2(y2)
    y3 = self.layer3(y3)
    return y1, y2, y3


class YOLOv3(nn.Module):
  def __init__(self, num_anchors=3, num_classes=20, pretrained: bool=False, weights: str='darknet53_256_c2ns-3aeff817.pth'):
    super().__init__()

    self.backbone = timm.create_model('darknet53',
      pretrained=pretrained, pretrained_cfg_overlay=dict(file=weights),
      features_only=True, out_indices=(-3, -2, -1))

    self.neck = Neck()
    self.head = Head(num_anchors, num_classes)

    if pretrained:
      self.neck.apply(init_weights)
      self.head.apply(init_weights)
    else:
      self.apply(init_weights)
  
  def forward(self, x):
    y1, y2, y3 = self.backbone(x)
    y1, y2, y3 = self.neck(y1, y2, y3)
    y1, y2, y3 = self.head(y1, y2, y3)
    return y1, y2, y3


if __name__ == '__main__':
  model = YOLOv3(pretrained=False)
  x = torch.randn(2, 3, 416, 416)
  y1, y2, y3 = model(x)
  print(f'y1.shape {y1.shape}')
  print(f'y2.shape {y2.shape}')
  print(f'y3.shape {y3.shape}')
