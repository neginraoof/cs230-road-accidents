import os
import numpy as np
from PIL import Image
from torch.utils import data
import cv2
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips


## ------------------- label conversion tools ------------------ ##
def labels2cat(label_encoder, list):
    return label_encoder.transform(list)


def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()


def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()


def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()


class MyVideoDataset(data.Dataset):
    def __init__(self, root, data_dirs, labels, n_frames=30, temporal_transform=None, spatial_transform=None):
        extensions = ('avi', 'mp4')
        print(data_dirs)
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.classes = data_dirs
        self.class_to_idx = {self.classes[i]: labels[i] for i in range(len(labels))}
        print(self.class_to_idx)
        self.samples = make_dataset(root, self.class_to_idx, extensions, is_valid_file=None)
        video_list = [x[0] for x in self.samples]
        self.video_clips = VideoClips(video_list,
                                      clip_length_in_frames=n_frames,
                                      frames_between_clips=n_frames,
                                      # frame_rate5
                                      )

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)

        if self.temporal_transform is not None:
            video = self.temporal_transform(video)
        if self.spatial_transform is not None:
            video = self.spatial_transform(video)

        labels = self.samples[video_idx][1]
        return video, labels

    def __len__(self):
        return self.video_clips.num_clips()
