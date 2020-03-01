from torch.utils import data
import os
import numpy
from torchvision.datasets.video_utils import VideoClips


class MyVideoDataset(data.Dataset):
    def __init__(self, root, data_dirs, labels, n_frames=30, temporal_transform=None, spatial_transform=None):
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        data_dirs = [os.path.join(root, d + ".avi") for d in data_dirs]
        self.labels = labels
        self.videos = data_dirs
        self.video_clips = VideoClips(self.videos,
                                      clip_length_in_frames=n_frames,
                                      frames_between_clips=n_frames,
                                      frame_rate=5
                                      )

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        if self.temporal_transform is not None:
            video = self.temporal_transform(video)
        if self.spatial_transform is not None:
            video = self.spatial_transform(video)

        label = self.labels[video_idx]
        return video, label

    def __len__(self):
        return self.video_clips.num_clips()
