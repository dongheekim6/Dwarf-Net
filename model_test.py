import torch
import torch.nn as nn
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F

#from res2net import res2net101_v1b_26w_4s  # assume this is implemented
model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net101_v1b_26w_4s(pretrained=False, **kwargs): 
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    return model

class Res2NetEncoder(nn.Module):
    def __init__(self, in_chans=3, pretrained=True):
        super().__init__()
        self.model = res2net101_v1b_26w_4s(pretrained=pretrained)
        if in_chans != 3:
        # 첫 번째 Conv2d 레이어 찾기
            if isinstance(self.model.conv1, nn.Sequential):
                old_conv = self.model.conv1[0]  # 첫 Conv2d
            else:
                old_conv = self.model.conv1     # Conv2d directly

            # 새로운 Conv2d 정의
            new_conv = nn.Conv2d(in_chans, old_conv.out_channels,
                                kernel_size=old_conv.kernel_size,
                                stride=old_conv.stride,
                                padding=old_conv.padding,
                                bias=False)
            
            # pretrained weight 평균 내서 복사
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            
            # Conv만 교체
            if isinstance(self.model.conv1, nn.Sequential):
                self.model.conv1[0] = new_conv
            else:
                self.model.conv1 = new_conv


    def forward(self, x):
        x0 = self.model.conv1(x)        # -> /2
        x0 = self.model.bn1(x0)
        x0 = self.model.relu(x0)
        x1 = self.model.maxpool(x0)     # -> /4
        x2 = self.model.layer1(x1)      # -> /4
        x3 = self.model.layer2(x2)      # -> /8
        x4 = self.model.layer3(x3)      # -> /16
        x5 = self.model.layer4(x4)      # -> /32
        return [x2, x3, x4, x5]  # multi-scale features
    
class DualRes2NetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.rgb_encoder = Res2NetEncoder(in_chans=3, pretrained=pretrained)
        self.gray_encoder = Res2NetEncoder(in_chans=1, pretrained=pretrained)

    def forward(self, x_rgb, x_gray):
        rgb_feats = self.rgb_encoder(x_rgb)
        gray_feats = self.gray_encoder(x_gray)
        return rgb_feats, gray_feats


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DualDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up4 = UpBlock(2048*2, 1024*2, 1024)
        self.up3 = UpBlock(1024, 512*2, 512)
        self.up2 = UpBlock(512, 256*2, 256)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, feats_rgb, feats_gray):
        # concat features at each stage
        f1 = torch.cat([feats_rgb[0], feats_gray[0]], dim=1)  # [256*2, 88, 88]
        f2 = torch.cat([feats_rgb[1], feats_gray[1]], dim=1)
        f3 = torch.cat([feats_rgb[2], feats_gray[2]], dim=1)
        f4 = torch.cat([feats_rgb[3], feats_gray[3]], dim=1)

        x = self.up4(f4, f3)
        x = self.up3(x, f2)
        x = self.up2(x, f1)
        x = self.up1(x)
        out = self.out_conv(x)
        return out

class DualRes2NetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DualRes2NetEncoder(pretrained=True)
        self.decoder = DualDecoder()

    def forward(self, x_rgb, x_gray):
        feats_rgb, feats_gray = self.encoder(x_rgb, x_gray)
        out = self.decoder(feats_rgb, feats_gray)
        out = F.interpolate(out, size=(352, 352), mode='bilinear', align_corners=True)
        return out  # (B, 1, H, W)


if __name__ == "__main__":
    model = DualRes2NetEncoder(pretrained=True)
    x_rgb = torch.randn(1, 3, 352, 352)
    x_gray = torch.randn(1, 1, 352, 352)

    rgb_feats, gray_feats = model(x_rgb, x_gray)

    for i, (rf, gf) in enumerate(zip(rgb_feats, gray_feats)):
        print(f"Stage {i+1} -> RGB: {rf.shape}, Gray: {gf.shape}")

    r_model = DualRes2NetUNet()
    result = r_model(x_rgb, x_gray)
    print('result', result.shape)