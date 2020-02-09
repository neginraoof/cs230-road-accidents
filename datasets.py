import torch
from torch.utils import data
import cv2
import os
import copy

def video_loader(video_file, channels=3, time_depth=5, x_size=240, y_size=256):
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    # nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = torch.FloatTensor(channels, time_depth, x_size, y_size)
    failedClip = False
    for f in range(time_depth):

        ret, frame = cap.read()
        if ret:
            frame = torch.from_numpy(frame)
            # HWC2CHW
            frame = frame.permute(2, 0, 1)
            frames[:, f, :, :] = frame

        else:
            print("Skipped!")
            failedClip = True
            break

    # for c in range(3):
    #     frames[c] -= self.mean[c]
    frames /= 255
    return frames, failedClip

def make_dataset(video_path, sample_duration):
    dataset = []

    n_frames = len(os.listdir(video_path))

    begin_t = 1
    end_t = n_frames
    sample = {
        'video': video_path,
        'segment': [begin_t, end_t],
        'n_frames': n_frames,
    }

    step = sample_duration
    for i in range(1, (n_frames - sample_duration + 1), step):
      sample_i = copy.deepcopy(sample)
      sample_i['frame_indices'] = list(range(i, i + sample_duration))
      sample_i['segment'] = torch.IntTensor([i, i + sample_duration - 1])
      dataset.append(sample_i)

    return dataset


class VideoDataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, video_path, sample_duration, distributed=False):
      'Initialization'
      self.dataset = make_dataset(video_path, sample_duration)

  def __len__(self):
      'Denotes the total number of samples'
      return len(self.dataset)

  def __getitem__(self, index):
      'Generates one sample of data'
      # Select sample
      path = self.data[index]['video']

      frame_indices = self.data[index]['frame_indices']
      # if self.temporal_transform is not None:
      # frame_indices = self.temporal_transform(frame_indices)
      clip = video_loader(path, frame_indices)
      # if self.spatial_transform is not None:
      # clip = [self.spatial_transform(img) for img in clip]
      # clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
      target = self.data[index]['segment']
      return clip, target

from torchvision.datasets.video_utils import VideoClips

class MyVideoDataset(data.Dataset):
    def __init__(self, video_paths):
        self.video_clips = VideoClips(video_paths,
                                      clip_length_in_frames=16,
                                      frames_between_clips=1,
                                      frame_rate=15)

    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        return video, audio
    
    def __len__(self):
        return self.video_clips.num_clips()
