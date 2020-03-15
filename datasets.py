from torch.utils import data
import os
from torchvision.datasets.video_utils import VideoClips
import transforms as T


class MyVideoDataset(data.Dataset):
    def __init__(self, root, data_dirs, labels, n_frames=30, fps=5, spatial_transform=None, temporal_transform=None, random_slice_size=0):
        data_dirs = [os.path.join(root, d + ".mp4") for d in data_dirs]
        self.videos = data_dirs
        self.labels = labels
        self.video_clips = VideoClips(self.videos,
                                      clip_length_in_frames=n_frames,
                                      frames_between_clips=n_frames,
                                      frame_rate=fps,
                                      num_workers=2
                                      )


        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.data_mean = None
        self.data_std = None
        self.random_slice_size = random_slice_size

    def set_stats(self, mean, std):
        self.data_mean, self.data_std = mean, std

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        if self.random_slice_size:
            video = T.RandomSlice(self.random_slice_size)(video)
        if self.temporal_transform is not None:
            video = self.temporal_transform(video)
        if self.spatial_transform is not None:
            video = self.spatial_transform(video)
        if self.data_mean is not None and self.data_std is not None:
            video = T.Normalize(mean=self.data_mean, std=self.data_std)(video)

        label = self.labels[video_idx]
        # print(video_idx, "---------------- ", self.video_clips.video_paths[video_idx], "------ label: ", label)
        return idx, video, label, video_idx

    def __len__(self):
        return self.video_clips.num_clips()
