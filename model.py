import torch
import torch.nn as nn
import os
import numpy as np


def conv3D_output_size(img_size, padding, kernel_size, stride):
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape


class Conv3dModel(nn.Module):
    def __init__(self, image_t_frames=120, image_height=90, image_width=120, drop_p=0.2, fc_hidden1=256, fc_hidden2=128):
        super(Conv3dModel, self).__init__()

        self.t_dim = image_t_frames
        self.image_height = image_height
        self.image_width = image_width

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.ch1, self.ch2 = 32, 48
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)

        self.conv1_outshape = conv3D_output_size((self.t_dim, self.image_height, self.image_width), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                      padding=self.pd1),
            nn.BatchNorm3d(self.ch1),
            nn.ReLU(inplace=True),
            # nn.Dropout3d(self.drop_p),
            nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                      padding=self.pd2),
            nn.BatchNorm3d(self.ch2),
            nn.ReLU(inplace=True),
            # nn.Dropout3d(self.drop_p),
            # MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
                      self.fc_hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_hidden1, self.fc_hidden2),
            nn.ReLU(inplace=True),
            # torch.nn.functional.dropout(x, p=self.drop_p, training=self.training)
            nn.Linear(self.fc_hidden2, 1),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x_3d):
        x = self.cnn_layers(x_3d)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
