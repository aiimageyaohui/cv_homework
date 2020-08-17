# -*- coding: utf-8 -*-
# @Time    : 2020/8/17 15:25
# @Author  : AiVision_YaoHui
# @FileName: vgg_fcn8s.py
from VGG_Extractor import VGG
import torch
import torch.nn as nn

class VGG_FCN8s(nn.Module):
    def __init__(self,classes):
        super(VGG_FCN8s, self).__init__()
        self.Extractor = VGG()

        self.fc_6 = nn.Conv2d(512, 4096, 7)
        self.relu_6 = nn.ReLU(inplace=True)
        self.drop_6 = nn.Dropout2d()

        self.fc_7 = nn.Conv2d(4096,4096,1)
        self.relu_7 = nn.ReLU(inplace=True)
        self.drop_7 = nn.Dropout2d()

        self.score_f5 = nn.Conv2d(4096, classes,1)
        self.score_f3 = nn.Conv2d(512, classes, 1)
        self.score_f4 = nn.Conv2d(256, classes, 1)


        # 上采样两倍
        self.ups_1 = nn.ConvTranspose2d(classes, classes,4, stride=2,bias=False)
        # 上采样四倍
        self.ups_2 = nn.ConvTranspose2d(classes,classes,4, stride=2, bias=False)
        # 上采样八倍
        self.ups_3 = nn.ConvTranspose2d(classes, classes,16, stride=8, bias=False)

    def forward(self, x):
        feature_extractor = self.Extractor(x)
        p3,p4,p5 = feature_extractor
        f6 = self.drop_6(self.relu_6(self.fc_6(p5)))
        f7 = self.score_f5(self.drop_7(self.relu_7(self.fc_7(f6))))
        ups1 = self.ups_1(f7)
        h = self.score_f4(p4)
        h = h[:,:,5:5 + ups1.size()[2],5:5 + ups1.size()[3]]
        h = h+ups1

        ups2 = self.ups_2(h)
        h = self.score_f3(p3)
        h = h[:, :, 9:9 + ups2.size()[2], 9:9 + ups2.size()[3]]
        h = h + ups2

        h = self.ups_3(h)
        final_scores = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return final_scores