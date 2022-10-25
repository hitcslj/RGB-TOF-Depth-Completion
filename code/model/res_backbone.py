import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )

class ConvBlock(nn.Module):
    def __init__(self, c, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv(c, c, 3, 1, 1),
            nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation),
        )
        self.relu = nn.PReLU(c)
        
    def forward(self, x):
        y = self.conv1(x)
        return self.relu(x + y)

class UNet(nn.Module):
    def __init__(self, in_planes, c=90):
        super(UNet, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            nn.Conv2d(c//2, c, 3, 2, 1),
        )
        self.convblock = torch.nn.ModuleList([])
        for i in range(4):
            self.convblock.append(ConvBlock(c, dilation=1))
        for i in range(4):
            self.convblock.append(ConvBlock(c, dilation=1))
        for i in range(4):
            self.convblock.append(ConvBlock(c, dilation=1))
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(2*c, c, 4, 2, 1),            
            nn.PReLU(c),
            conv(c, c, 3, 1, 1),
            nn.ConvTranspose2d(c, c, 4, 2, 1),
            nn.PReLU(c),
            nn.Conv2d(c, 1, 3, 1, 1)
        )

    # flip is used when inference
    def forward(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']
        bz = dep.shape[0]
        dep_max = torch.max(dep.view(bz,-1),1, keepdim=False)[0].view(bz,1,1,1)
        dep = dep/(dep_max+1e-6)
        x = torch.cat((rgb,dep),dim=1)
        x1 = x.flip(3)
        def func(x):
            feat = self.conv0(x)
            tmp = feat
            for i in range(12):
                feat = self.convblock[i](feat)
            y = self.lastconv(torch.cat((feat, tmp), 1))
            y = torch.sigmoid(y) * dep_max
            return y
        return (func(x)+func(x1).flip(3))/2

     
    # def forward(self, sample):
    #     rgb = sample['rgb']
    #     dep = sample['dep']
    #     bz = dep.shape[0]
    #     dep_max = torch.max(dep.view(bz,-1),1, keepdim=False)[0].view(bz,1,1,1)
    #     dep = dep/(dep_max+1e-6)
    #     x = torch.cat((rgb,dep),dim=1)
    #     feat = self.conv0(x)
    #     tmp = feat
    #     for i in range(12):
    #         feat = self.convblock[i](feat)
    #     y = self.lastconv(torch.cat((feat, tmp), 1))
    #     y = torch.sigmoid(y) * dep_max
    #     return y
        