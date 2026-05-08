import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # ceil_mode=True 避免奇数尺寸丢失
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv, 自动对齐通道"""
    def __init__(self, x1_channels, x2_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(x1_channels, x1_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(x1_channels + x2_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 对齐大小
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UNetAnySize(nn.Module):
    """任意输入尺寸，输出同尺寸"""
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        # 注意这里传入上采样前和 skip connection 的通道数
        self.up1 = Up(512, 256, 256, bilinear)
        self.up2 = Up(256, 128, 128, bilinear)
        self.up3 = Up(128, 64, 64, bilinear)
        self.up4 = Up(64, 32, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# 测试任意尺寸输入
if __name__ == "__main__":
    model = UNetAnySize(n_channels=1, n_classes=1)
    for H, W in [(512,32), (127,129), (33,50), (64,101)]:
        x = torch.randn(2, 1, H, W)
        y = model(x)
        print(f"Input: {x.shape}, Output: {y.shape}")
