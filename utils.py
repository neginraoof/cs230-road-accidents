import argparse
import moviepy.editor
import numpy as np
import torch

parser = argparse.ArgumentParser(description="Video Classification")
parser.add_argument('--pretrained', default=False, action="store_true")
parser.add_argument('--crop_videos', default=False, action="store_true")


def crop_video(data_dirs):
    outputs = []
    durations = []
    for i in data_dirs:
        # import to moviepy
        clip = moviepy.editor.VideoFileClip("./video_data/{}.avi".format(i))
        print("{}: Duration  {} sec".format(i, clip.duration))
        durations.append(clip.duration)
    print("Min duration is", min(durations))

    for i in data_dirs:
        # import to moviepy
        clip = moviepy.editor.VideoFileClip("./video_data/{}.avi".format(i))
        print("{}: Duration  {} sec".format(i, clip.duration))
        # select a random time point
        length = min(durations)
        start = round(np.random.uniform(0, clip.duration - length), 2)
        # cut a subclip
        out_clip = clip.subclip(start, start + length)
        out_clip.write_videofile("./video_data_clip/{}.mp4".format(i), audio_codec='aac')
        outputs.append(out_clip)

    return outputs


def get_stats(data_loader):
    N_count = 0
    inputs = ()
    for batch_idx, (X, _, __) in enumerate(data_loader):
        X = X.to(dtype=torch.float32)
        print("x", X.shape)
        N_count += X.shape[0]
        inputs += (X,)
    in_cat = torch.cat(inputs, 0)

    mean = in_cat.mean(dim=[0, 2, 3, 4])
    std = in_cat.std(dim=[0, 2, 3, 4])
    print("Mean ", mean)
    print("Std ", std)
    return mean, std
