'''
This file is the model in "A Unified Deep Learning Framework for Urban Functional Zone Extraction Based on Multi-source Heterogeneous Data"
'''
import torch.nn as nn
from torchvision import models
import torch.nn.functional as f
import torch


# ======part one======
# residual block
class ResidualBlock(nn.Module):
    #实现子module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=True),
            nn.Dropout2d(),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),  # inplace = True原地操作
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=True),
            nn.Dropout2d(),
            nn.BatchNorm2d(out_ch))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return f.relu(out)


# resnet50
# the output is mapped to 2048-d
class ResNet50(nn.Module):  # 224x224x3
    #实现主module:ResNet34
    def __init__(self, in_channels=3):
        super(ResNet50, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))

        self.layer1 = self.make_layer(64, 64, 3)  # 56x56x64
        self.layer2 = self.make_layer(64, 128, 4, stride=2)
        self.layer3 = self.make_layer(128, 256, 6, stride=2)
        self.layer4 = self.make_layer(256, 512, 3, stride=2)
        self.conv = nn.Sequential(nn.Conv2d(512, 2048, 1),
                                  nn.BatchNorm2d(2048), nn.ReLU())

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
            nn.BatchNorm2d(out_ch))
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):  # 224x224x3
        x = self.pre(x)  # 56x56x64
        x = self.layer1(x)  # 56x56x64
        x = self.layer2(x)  # 28x28x128
        x = self.layer3(x)  # 14x14x256
        x = self.layer4(x)  # 7x7x512
        x = self.conv(x)
        return x


# This is the first part: complementary feature learning and fusing
class complementary_fusion(nn.Module):
    def __init__(self, num_classes, need_feature=False, num_poi_layer=8, r=16):
        '''
        Parameters:
         num_class: total number of class in your classification category system
         need_feature: whether the our put contains extracted features?
         num_poi_layer: total number of class in your POI category system
         r: rescale ratio
        Note:
         When need_feature is "False", only this model will only return the prediction
         When it is "True", this model will return a tuple : (visual feature, social feature, fused feature, prediction)
        '''
        super(complementary_fusion, self).__init__()
        self.need_feature = need_feature
        self.model_name = 'complementary_fusion'
        self.img_encoder = ResNet50(in_channels=3)
        self.poi_encoder = ResNet50(in_channels=num_poi_layer)
        self.LWM = nn.Sequential(
            nn.Conv2d(num_poi_layer, num_poi_layer * r, 1, 1, 0),
            nn.BatchNorm2d(num_poi_layer * r), nn.ReLU(),
            nn.Conv2d(num_poi_layer * r, num_poi_layer, 1, 1, 0),
            nn.BatchNorm2d(num_poi_layer), nn.Sigmoid())
        self.FAFS = nn.Sequential(nn.Conv2d(4096, 4096 // r, 1, 1, 0),
                                  nn.BatchNorm2d(4096 // r), nn.ReLU(),
                                  nn.Conv2d(4096 // r, 2, 1, 1, 0),
                                  nn.BatchNorm2d(2), nn.Sigmoid())
        self.out = nn.Conv2d(2048, num_classes, 1, 1, 0)

    def forward(self, img, poi):
        '''
        Parameters:
         img: a batch of image tensor (size: B*C*H*W)
         poi: a batch of distance heatmap tensor (size: B*C*H*W)
        Return:
         Ref: notes in __inti__()
        '''
        img = self.img_encoder(img)
        img = f.adaptive_avg_pool2d(img, (1, 1))
        b, _, m, n = poi.size()
        u = f.adaptive_avg_pool2d(poi, (1, 1))
        poi_weight = self.LWM(u)
        poi_weight = f.interpolate(poi_weight, (m, n), mode='nearest')
        poi *= poi_weight
        poi = self.poi_encoder(poi)
        poi = f.adaptive_avg_pool2d(poi, (1, 1))
        feature_weight = self.FAFS(torch.cat([img, poi], 1))
        fuse_feature = img * feature_weight[:, 0, :, :].unsqueeze(
            1) + poi * feature_weight[:, 1, :, :].unsqueeze(1)
        out = self.out(fuse_feature)
        if self.need_feature:
            return img, poi, fuse_feature, out
        else:
            return out


#======part two======
# the following two models (ResNet_block, brnnnet) are for two kinds of spatial information modeling
class ResNet_block(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernal_size=3,
                 padding=1):
        super(ResNet_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, 1, padding=1),
            nn.Dropout2d(), nn.BatchNorm2d(input_channels), nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, padding=1),
            nn.Dropout2d(), nn.BatchNorm2d(output_channels), nn.ReLU())
        self.bn = nn.Sequential(nn.BatchNorm2d(output_channels), nn.ReLU())

    def forward(self, x):
        input_ = x
        x = self.conv1(x)
        x = self.conv2(x)
        # input_ = self.conv3(input_)
        return self.bn(x + input_)


class brnnnet_layer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(brnnnet_layer, self).__init__()
        if output_channels % 2 != 0:
            output_channels += 1
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.Vlayer = nn.GRU(input_channels,
                             int(input_channels / 2),
                             1,
                             bidirectional=True)
        self.Hlayer = nn.GRU(input_channels,
                             int(output_channels / 2),
                             1,
                             bidirectional=True)

    def forward(self, x):
        b, c, m, n = x.size()
        x = x.permute(2, 3, 0, 1)
        V_map = torch.zeros(m, n, b, self.input_channels).cuda()
        H_map = torch.zeros(m, n, b, self.output_channels).cuda()
        for i in range(0, n):
            V_map[:, i, :, :], _ = self.Vlayer(x[:, i, :, :])
        for i in range(0, m):
            H_map[i, :, :, :], _ = self.Hlayer(V_map[i, :, :, :])
        return H_map.permute(2, 3, 0, 1)


# cross transfer unit
# the input and output channel numbers are required
class CTU(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CTU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 3, 1, padding=1),
            nn.Dropout2d(), nn.BatchNorm2d(input_channels), nn.ReLU())

        self.brnnnet1 = brnnnet_layer(input_channels, input_channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 1, padding=1),
            nn.Dropout2d(), nn.BatchNorm2d(output_channels), nn.ReLU())

        self.brnnnet2 = brnnnet_layer(input_channels, output_channels)

        self.bn = nn.Sequential(nn.BatchNorm2d(output_channels), nn.ReLU())

    def forward(self, x):
        input_ = x
        x1 = self.conv1(x)
        x1 = self.brnnnet2(x1)
        x2 = self.brnnnet1(x)
        x2 = self.conv2(x2)
        return self.bn(x1 + x2 + input_)


# saptial information modeling
class spatial_information_modeling(nn.Module):
    def __init__(self,
                 input_feature_map_channnels=2048,
                 num_classes=10,
                 num_CTU=3,
                 num_FCN_layers=1):
        '''
        Parameters:
         input_feature_map_channnels: the channel number C of input tensor (B*C*H*W)
         num_classes: total number of class in your classification category system
         num_CTU: number of CTUs
         num_FCN_layers: number of fully connected layer
        '''
        super(spatial_information_modeling, self).__init__()
        self.num_classes = num_classes
        self.conv1x1 = nn.Sequential(
            nn.BatchNorm2d(input_feature_map_channnels), nn.ReLU(),
            nn.Conv2d(input_feature_map_channnels, 512, 1), nn.Dropout2d(),
            nn.BatchNorm2d(512), nn.ReLU())
        self.CTU = []
        for i in range(num_CTU):
            self.CTU += [CTU(512, 512)]
        self.CTU = nn.Sequential(*self.CTU)
        self.out = nn.Conv2d(512, num_classes, 1, 1)

    def forward(self, x):
        '''
        Parameters:
         x: the input feature extracted in the first stage
        Return:
         prediction result
        '''
        x = self.conv1x1(x)
        x = self.CTU(x)
        return self.out(x)
