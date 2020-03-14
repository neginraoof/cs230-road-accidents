import argparse
import moviepy.editor
import numpy as np
import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from os import path


parser = argparse.ArgumentParser(description="Video Classification")
parser.add_argument('--pretrained', default=False, action="store_true")
parser.add_argument('--crop_videos', default=False, action="store_true")
parser.add_argument('--get_stats', default=False, action="store_true")
parser.add_argument('--resume', default=False, action="store_true")

def read_data_labels(path_to_csv, categories):
    # Load Train Data
    df = pd.read_csv(path_to_csv)
    video_ids = os.listdir("./video_data")
    video_ids = [vid_id.replace('.avi', '') for vid_id in video_ids]
    #video_ids = [vid_id.replace('.mp4', '') for vid_id in video_ids]
    data_subset = df.loc[df['Video ID'].isin(video_ids)]
    video_ids = data_subset['Video ID'].to_numpy()
    data_list = video_ids
    data_label = data_subset['Label'].to_numpy()
    # Transform labels to categories
    data_label = np.rint(data_label)
    danger_category = np.asarray(categories).reshape(-1,)
    label_encoder = LabelEncoder()
    # print(danger_category)
    label_encoder.fit(danger_category)
    data_label = label_encoder.transform(data_label.reshape(-1,))
    # TODO: Fix these lines to get all videos from CSV file
    print("data list ", data_list)
    return data_list, data_label


def crop_video(data_dirs):
    outputs = []
    durations = []
    for i in data_dirs:
        # import to moviepy
        if path.exists("./video_data/{}.avi".format(i)):
            clip = moviepy.editor.VideoFileClip("./video_data/{}.avi".format(i))
            print("{}: Duration  {} sec".format(i, clip.duration))
            durations.append(clip.duration)
    print("Min duration is", min(durations))

    for i in data_dirs:
        # import to moviepy
        if path.exists("./video_data/{}.avi".format(i)):
            clip = moviepy.editor.VideoFileClip("./video_data/{}.avi".format(i))
            print("{}: Duration  {} sec".format(i, clip.duration))
            # select a random time point
            length = min(durations)
            start = round(np.random.uniform(0, clip.duration - length), 2)
            # cut a subclip
            out_clip = clip.subclip(start, start + length)
            if not path.exists("./video_data_clip/{}.mp4".format(i)):
                out_clip.write_videofile("./video_data_clip/{}.mp4".format(i), audio_codec='aac')
            outputs.append(out_clip)

    return outputs


def get_stats(data_loader, device="cpu"):
    N_count = 0
    mean_s = ()
    var_s = ()
    for batch_idx, (___, X, _, __) in enumerate(data_loader):
        X = X.to(dtype=torch.float32)
        N_count += X.shape[0]
        mean_s += (X.mean(dim=[0, 2, 3, 4]),)
        var_s += (X.var(dim=[0, 2, 3, 4]),)
        print("next")
 
    torch.device(device)
    mean_cat = torch.stack(mean_s)
    var_cat = torch.stack(var_s)
    
    mean = mean_cat.mean(dim=0)
    var = var_cat.mean(dim=0)
    std = torch.sqrt(var)
    print("Mean ", mean)
    print("Std ", std)
    return mean, std
