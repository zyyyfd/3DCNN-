

import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F





class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4,c5,**kwargs):
        super(Inception, self).__init__(**kwargs)
        
        self.p1_1 = nn.Conv3d(in_channels, c1,stride=1, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(c1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        
        self.p2_1 = nn.Conv3d(in_channels, c2[0],stride=1, kernel_size=1)
        self.p2_2 = nn.Conv3d(c2[0], c2[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(c2[1])
        self.relu2 = nn.LeakyReLU(inplace=True)
        
        self.p3_1 = nn.Conv3d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv3d(c3[0], c3[1], kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm3d(c3[1])
        self.relu3 = nn.LeakyReLU(inplace=True)
        
        self.p4_1 = nn.Conv3d(in_channels, c4[0], kernel_size=1)
        self.p4_2 = nn.Conv3d(c4[0], c4[1], kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm3d(c4[1])
        self.relu4 = nn.LeakyReLU(inplace=True)
       
        self.p5_1 = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
        self.p5_2 = nn.Conv3d(in_channels, c5,kernel_size=1)
        self.bn5 = nn.BatchNorm3d(c5)
        self.relu5 = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        p1=self.p1_1(x)
        p1=self.bn1(p1)
        p1=self.relu1(p1)
        
        p2=self.p2_1(x)
        p2=self.p2_2(p2)
        p2=self.bn2(p2)
        p2=self.relu2(p2)
        
        p3=self.p3_1(x)
        p3=self.p3_2(p3)
        p3=self.bn3(p3)
        p3=self.relu3(p3)
        
        p4=self.p4_1(x)
        p4=self.p4_2(p4)
        p4=self.bn4(p4)
        p4=self.relu4(p4)
        
        p5=self.p5_1(x)
        p5=self.p5_2(p5)
        p5=self.bn5(p5)
        p5=self.relu5(p5)
        
        p=torch.cat((p1,p2),dim=1)
        p=torch.cat((p,p3),dim=1)
        p=torch.cat((p,p4),dim=1)
        p=torch.cat((p,p5),dim=1)
        
       
        return p




class CABlock(nn.Module):
    def __init__(self, channel, h,w,l, reduction=16):
        super(CABlock, self).__init__()
        
        self.h = h
        self.w = w
        self.l = l
        
        self.avg_pool_x = nn.AdaptiveAvgPool3d((h, 1,1))
        self.avg_pool_y = nn.AdaptiveAvgPool3d((1, w,1))
        self.avg_pool_l = nn.AdaptiveAvgPool3d((1, 1,l))
        
        self.conv_1x1x1 = nn.Conv3d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)
        self.relu = nn.LeakyReLU()
        self.bn = nn.BatchNorm3d(channel // reduction)
        self.F_h = nn.Conv3d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        
        self.F_w = nn.Conv3d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        
        self.F_l = nn.Conv3d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
        self.sigmoid_l = nn.Sigmoid()
    def forward(self, x):
       
        x_h = self.avg_pool_x(x).permute(0, 1, 4, 3, 2)
        
        x_w = self.avg_pool_y(x).permute(0,1,4,2,3)
       
        x_l = self.avg_pool_l(x)
        
        x_cat_conv_relu = self.relu(self.conv_1x1x1(torch.cat((x_h, x_w,x_l), 4)))
        
        x_cat_conv_split_h, x_cat_conv_split_w,x_cat_conv_split_l  = x_cat_conv_relu.split([self.h, self.w,self.l], 4)#拆分卷积后的维度
       
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1,4,3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w.permute(0,1,2,4,3)))
        s_l = self.sigmoid_l(self.F_l(x_cat_conv_split_l))
        
        out = x * s_h.expand_as(x) * s_w.expand_as(x)*s_l.expand_as(x)
        return out





class CNN3D(nn.Module):
    def __init__(self):
        super(model13D,self).__init__()
        self.b1 = nn.Sequential(nn.Conv3d(13,256,kernel_size=1),
                                nn.BatchNorm3d(256),
                                nn.LeakyReLU(inplace=True))
                                
        self.p1=nn.Sequential(Inception(256,32,(32,64),(32,64),(16,32),64),CABlock(256,32,32,32))
        self.p2=nn.Sequential(Inception(256,128,(16,32),(16,32),(16,32),32),CABlock(256,32,32,32))
        self.p3=nn.Sequential(Inception(256,32,(64,128),(16,32),(16,32),32),CABlock(256,32,32,32))
        self.p4=nn.Sequential(Inception(256,64,(32,64),(32,64),(16,32),32),CABlock(256,32,32,32))
                                
           
        self.b2 = nn.Sequential(nn.Conv3d(256,512,kernel_size=1),
                                nn.BatchNorm3d(512),
                                nn.LeakyReLU(inplace=True),
                                nn.AvgPool3d(kernel_size=3,stride=2,padding=1))
            
        self.b3 = nn.Sequential(nn.Conv3d(512,1024,kernel_size=1),
                                nn.BatchNorm3d(1024),
                                nn.LeakyReLU(inplace=True),
                                nn.AvgPool3d(kernel_size=3,stride=2,padding=1))
        
        self.b4 = nn.Sequential(nn.Conv3d(1024,2048,kernel_size=3,stride=2,padding=1),
                                nn.BatchNorm3d(2048),
                                nn.LeakyReLU(inplace=True),
                                nn.AvgPool3d(kernel_size=3,stride=2,padding=1))
        
        self.b5 = nn.Sequential(nn.Flatten(),
                                nn.Linear(2048*8,2048),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(2048,1024),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(1024,256),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(256,64),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(64,16),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(16,1))
    def forward(self, x):
        x=self.b1(x)
        x=self.p1(x)
        x=self.p2(x)
        x=self.p3(x)
        x=self.p4(x)
        x=self.b2(x)
        x=self.b3(x)
        x=self.b4(x)
        x=self.b5(x)
        x=x.view(1)
        return x

