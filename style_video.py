import functools
import io
import os
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from fast_neural_style_pytorch.stylize import stylize_folder


# VIDEO PROCESSING =====================================================
def slice_frames(video_file, frame_save_folder, frame_name, skip_count=1):
    """ Slices a video into its frames and saves the result in test_frames/
    Args
        video_file (str): path name of video to slice up
    Output
        Returns the number of frames in the video
    """
    cap = cv2.VideoCapture(video_file)

    framecount = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret and not (framecount % skip_count):
            filename = os.path.join(frame_save_folder, f"{frame_name}{framecount}.jpg")
            cv2.imwrite(filename, frame)
            framecount += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return framecount


def combine_frames(frame_paths, output_video_folder, output_filename):
    """images -> frames"""

    frame = cv2.imread(frame_paths[0])
    height, width, layers = frame.shape

    fourcc_codec = cv2.VideoWriter_fourcc(*"avc1")
    fps = 25
    output_path = os.path.join(output_video_folder, output_filename)
    video = cv2.VideoWriter(output_path, fourcc_codec, fps, (width, height))

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    return output_path


def fast_style_transfer_video_file(content_video_path, style_path):
    """
    Video style transfer for a given content video file fname

    Args
        - content_video_path (str): filepath of content video
        - style_path (str): filepath of pretrained style network .pth file
    """
    output_folder = "static"
    video_filename = "fast_output.mp4"
    start_time = time.time()
    n_frames = slice_frames(
        content_video_path, 
        frame_save_folder="fast_frames/content_folder", 
        frame_name="testframe"
    )

    print(f"Time to slice up {time.time() - start_time} for {n_frames} frames")

    start_time = time.time()

    content_frame_save_path = "fast_frames"
    style_frame_save_path = "fast_output_frames"
    frame_paths = stylize_folder(style_path, content_frame_save_path, style_frame_save_path, batch_size=1)

    print(f"Time to style transfer {time.time() - start_time}")

    start_time = time.time()
    output_path = combine_frames(frame_paths, output_folder, video_filename)
    print(f"Time to combine {time.time() - start_time}")

    return video_filename



if __name__ == "__main__":
    # frame_paths = [f"output_frames/outputframe{i}.jpg" for i in range(50)]
    # combine_frames(frame_paths, "output_videos", "output.mp4")
    style_transfer_video_lite(48, style_image_path="static/udnie.jpg")
    # style_transfer_video_file("trimmed.mp4", "static/udnie.jpg")
