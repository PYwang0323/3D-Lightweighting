'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'], [512, 512, 512, 'M', 512, 512, 512, 'M']],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG3d(nn.Module):
    def __init__(self, vgg_name, num_classes):
        super(VGG3d, self).__init__()
        # experiments about different Depth 
        # eg:when original channel = 512
        # we set depth = 8 or 16 or 32 or 64 
        self.Depth = 8

        self.features = self._make_layers(cfg[vgg_name][0])
        self.features3d = self._make_layers3d(cfg[vgg_name][1])
        self.avgpool= nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, num_classes)

    def trans2to3(self,x):
        N, C, H, W = x.size()
        x = x.unsqueeze(1)
        x = x.view(N, int(C/self.Depth), self.Depth, H, W)
        return x

    def trans3to2(self,x):
        N, C, D, H, W = x.size()
        x = x.view(N, C*D, H, W)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def _make_layers3d(self, cfg):
        layers = []
        in_channels = int(256 /self.Depth)
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))]
            else:
                x = int(x/self.Depth)  
                layers += [nn.Conv3d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm3d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        # layers += [nn.AdaptiveAvgPool2d(1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)

        # pdb.set_trace()
        
        out = self.trans2to3(out)
        out = self.features3d(out)

        # pdb.set_trace()

        out = self.trans3to2(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())
# test()


# print("VGG-16_3d_L2_D8")

# net = VGG3d('VGG16', 10)

# params = list(net.parameters())
# k = 0
# for i in params:
#     l = 1
#     print("该层的结构：" + str(list(i.size())))
#     for j in i.size():
#         l *= j
#     print("该层参数和：" + str(l))
#     k = k + l
# print("总参数数量和：" + str(k))



