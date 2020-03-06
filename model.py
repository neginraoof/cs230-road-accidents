import torch
import torch.nn as nn
import os
import numpy as np
import torchvision


def output_size_3d(img_size, padding, kernel_size, stride):
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape


class Conv3dModel(nn.Module):
    def __init__(self, image_t_frames=120, image_height=300, image_width=400, drop_p=0.2, fc_hidden1=256, fc_hidden2=128,
                 num_classes=4):
        super(Conv3dModel, self).__init__()

        self.t_dim = image_t_frames
        self.image_height = image_height
        self.image_width = image_width

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.ch1, self.ch2 = 32, 64
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)
        self.num_classes = num_classes

        self.conv1_outshape = output_size_3d((self.t_dim, self.image_height, self.image_width), self.pd1, self.k1, self.s1)
        self.conv2_outshape = output_size_3d(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.max_outshape = output_size_3d(self.conv2_outshape, (0, 0, 0), (2, 2, 2), (2, 2, 2))

        self.cnn_layers = nn.Sequential(
            # First 3D convolution layer + BN + ReLU (dropout optional)
            nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                      padding=self.pd1),
            nn.BatchNorm3d(self.ch1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p),
            # Second 3D convolution layer + BN + ReLU (dropout optional)
            nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                      padding=self.pd2),
            nn.BatchNorm3d(self.ch2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Dropout3d(self.drop_p),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(self.ch2 * self.max_outshape[0] * self.max_outshape[1] * self.max_outshape[2],
                      self.fc_hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.drop_p),
            nn.Linear(self.fc_hidden2, self.num_classes),
        )

    def forward(self, x_3d):
        x = self.cnn_layers(x_3d)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    
class ResNet18(nn.Module):
    def __init__(self, num_classes=4, drop_p=0.2, fc_hidden1=400, fc_hidden2=100):
        super(ResNet18, self).__init__()
        
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.num_classes = num_classes
 
        self.resnet_layers = torchvision.models.video.r3d_18(pretrained=True, progress=True)        
        self.linear_layers = nn.Sequential(
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.drop_p),
            nn.Linear(self.fc_hidden2, self.num_classes),
        )

    def forward(self, x_3d):
        x = self.resnet_layers(x_3d)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


class Conv3dModelOrdinal(nn.Module):
    def __init__(self, image_t_frames=120, image_height=300, image_width=400, drop_p=0.2, fc_hidden1=256, fc_hidden2=128,
                 num_classes=4):
        super(Conv3dModelOrdinal, self).__init__()

        self.t_dim = image_t_frames
        self.image_height = image_height
        self.image_width = image_width

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.ch1, self.ch2 = 32, 64
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)
        #self.num_classes = num_classes
        self.softmax = torch

        self.conv1_outshape = output_size_3d((self.t_dim, self.image_height, self.image_width), self.pd1, self.k1, self.s1)
        self.conv2_outshape = output_size_3d(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.max_outshape = output_size_3d(self.conv2_outshape, (0, 0, 0), (2, 2, 2), (2, 2, 2))

        self.cnn_layers = nn.Sequential(
            # First 3D convolution layer + BN + ReLU (dropout optional)
            nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                      padding=self.pd1),
            nn.BatchNorm3d(self.ch1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.drop_p),
            # Second 3D convolution layer + BN + ReLU (dropout optional)
            nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                      padding=self.pd2),
            nn.BatchNorm3d(self.ch2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Dropout3d(self.drop_p),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(self.ch2 * self.max_outshape[0] * self.max_outshape[1] * self.max_outshape[2],
                      self.fc_hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=self.drop_p),

        )

    def forward(self, x_3d):
        x = self.cnn_layers(x_3d)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        lin_1 = nn.Linear(self.fc_hidden2, 1)(x)
        sigma_1 = torch.nn.LogSoftmax()(lin_1)
        lin_2 = nn.Linear(self.fc_hidden2, 1)(x)
        sigma_2 = torch.nn.LogSoftmax()(lin_2)
        lin_3 = nn.Linear(self.fc_hidden2, 1)(x)
        sigma_3 = torch.nn.LogSoftmax()(lin_3)

        r_1 = sigma_1
        r_2 = 1 - sigma_1
        r_3 = (1 - sigma_1) * sigma_2
        r_4 = (1 - sigma_1) * (1 - sigma_2)
        r_5 = (1 - sigma_1) * (1 - sigma_2) * sigma_3
        r_6 = (1 - sigma_1) * (1 - sigma_2) * (1 - sigma_3)
        out = torch.stack((r_1, r_2, r_3, r_4, r_5, r_6), dim=1)

        print(r_1.shape, " out shape ", out.shape)
        print(out)
        return out