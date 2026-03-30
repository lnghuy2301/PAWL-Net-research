import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class PhysicsSwinUNet(nn.Module):
    def __init__(self, in_channels=10, out_channels=1, img_size=128):
        super().__init__()
        self.encoder = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, features_only=True, img_size=img_size)

        old_embed = self.encoder.patch_embed.proj
        new_embed = nn.Conv2d(in_channels, old_embed.out_channels, kernel_size=old_embed.kernel_size, stride=old_embed.stride, padding=old_embed.padding)
        with torch.no_grad():
            new_embed.weight[:, :3, :, :] = old_embed.weight
            nn.init.xavier_uniform_(new_embed.weight[:, 3:, :, :])
        self.encoder.patch_embed.proj = new_embed

        enc_chs = self.encoder.feature_info.channels()
        self.bottleneck = nn.Sequential(nn.Conv2d(enc_chs[-1], enc_chs[-1], 3, padding=1), nn.BatchNorm2d(enc_chs[-1]), nn.ReLU())
        self.up1 = UpBlock(enc_chs[-1], enc_chs[-2])
        self.up2 = UpBlock(enc_chs[-2], enc_chs[-3])
        self.up3 = UpBlock(enc_chs[-3], enc_chs[-4])
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            nn.Conv2d(enc_chs[-4], 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x):
        f1, f2, f3, f4 = self.encoder(x)
        f1, f2, f3, f4 = f1.permute(0,3,1,2), f2.permute(0,3,1,2), f3.permute(0,3,1,2), f4.permute(0,3,1,2)
        x = self.bottleneck(f4)
        x = self.up1(x, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)
        return torch.sigmoid(self.final_up(x))

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y)**2 + self.eps**2))

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([[0, 1, 0], [1, -4, 1],[0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('k', k)
        self.loss = CharbonnierLoss()
    def forward(self, x, y):
        edge_x = F.conv2d(x, self.k, padding=1)
        edge_y = F.conv2d(y, self.k, padding=1)
        return self.loss(edge_x, edge_y)

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num_losses=2):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(num_losses))
    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            precision = 0.5 * torch.exp(-self.params[i])
            loss_sum += precision * loss + 0.5 * self.params[i]
        return loss_sum