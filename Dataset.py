import torch
from torch.utils import data

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, video_list, labels):
        'Initialization'
        self.labels = labels
        self.video_list = video_list

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.video_list)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.video_list[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y