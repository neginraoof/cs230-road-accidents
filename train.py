import numpy as np
import os
import torch
# import torchvision
# import argparse
# import cv2
# from datasets import MyVideoDataset
# import transforms as T
# from torch.autograd import Variable
# from torchvision.datasets.samplers import DistributedSampler, UniformClipSampler, RandomClipSampler
# from torch.utils.data.dataloader import default_collate
# from utils import AverageMeter
# from model import CNN3D
from sklearn.metrics import accuracy_score
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
# import matplotlib.pyplot as plt


def train_one_epoch(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    model.train()

    losses = []
    scores = []
    N_count = 0  # counting total trained sample in one epoch

    for batch_idx, (X, y) in enumerate(train_loader):
        # # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = model(X)  # output size = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output

        print("preeeeeeeeeeeeeeeeeeeeeeeed ", y_pred)
        print("issssssssssssssssssssssssss ", y)

        step_score = accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())
        scores.append(step_score)  # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),
                100 * step_score))

    return losses, scores
