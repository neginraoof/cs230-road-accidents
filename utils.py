import argparse
import moviepy.editor
import numpy as np


parser = argparse.ArgumentParser(description="Video Classification")
parser.add_argument('--pretrained', default=False, action="store_true")


def crop_video(data_dirs):
    outputs = []
    durations = []
    for i in data_dirs:
        # import to moviepy
        clip = moviepy.editor.VideoFileClip("./video_data/{}.avi".format(i))
        print(i, ": Duration   ", clip.duration / 60)
        durations.append(clip.duration)
    print("Min duration is", min(durations))

    for i in data_dirs:
        # import to moviepy
        clip = moviepy.editor.VideoFileClip("./video_data/{}.avi".format(i))
        print(i, ": Duration   ", clip.duration / 60)
        # select a random time point
        length = min(durations)
        start = round(np.random.uniform(0, clip.duration - length), 2)
        # cut a subclip
        out_clip = clip.subclip(start, start + length)
        out_clip.write_videofile("./video_data_clip/{}.mp4".format(i), audio_codec='aac')
        outputs.append(out_clip)

    return outputs