import functools
import io
import os
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


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
    """Combines frames into a video"""

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


def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape
    )
    return image


@functools.lru_cache(maxsize=None)
def load_image(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""

    # deal with possible RGBA vs RGB issues
    png = Image.open(image_path).convert("RGBA")
    background = Image.new("RGBA", png.size, (255, 255, 255))

    img = Image.alpha_composite(background, png).convert("RGB")
    img = np.array([np.asarray(img)])

    if img.max() > 1.0:
        img = img / 255.0
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img


def preprocesses_style_image(style_image_path=None):
    style_img_size = (256, 256)
    if not style_image_path:
        style_image_url = "https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg"
        style_image_path = tf.keras.utils.get_file(
            os.path.basename(style_image_url)[-128:], style_image_url
        )

    style_image = load_image(style_image_path, style_img_size)
    style_image = tf.nn.avg_pool(
        style_image, ksize=[3, 3], strides=[1, 1], padding="SAME"
    )

    return style_image


def get_image_path_from_url(image_url):
    image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
    return image_path


def get_content_image_from_path(content_image_path):
    output_image_size = 384
    content_img_size = (output_image_size, output_image_size)
    content_image = load_image(content_image_path, content_img_size)

    return content_image


# TF Lite Functions
def run_style_transform(
    style_bottleneck, preprocessed_content_image, style_transform_path
):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_transform_path)

    # Set model input.
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()

    # Set model inputs.
    interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()

    # Transform content image.
    stylized_image = interpreter.tensor(interpreter.get_output_details()[0]["index"])()

    return stylized_image


# Function to run style prediction on preprocessed style image.
def run_style_predict(preprocessed_style_image, style_predict_path):
    # Load the model.
    interpreter = tf.lite.Interpreter(model_path=style_predict_path)

    # Set model input.
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)

    # Calculate style bottleneck.
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(
        interpreter.get_output_details()[0]["index"]
    )()

    return style_bottleneck


def get_style_transfer(
    content_image, nframe, style_image, use_tflite=False, send_image=False
):
    """
    Performs style transfer on a content image and style image. This function is intended for use when styling one single image,
    not a video. If send_image is True, will return the pillow image object itself, otherwise will save to output_frames.
    """
    try:
        fin = open("path_info.txt", "r+")
        path = fin.readline().strip()
        if len(path) > 0:
            os.environ["TFHUB_CACHE_DIR"] = path  # Any folder that you can access
    except:
        pass

    if use_tflite:
        style_predict_path = tf.keras.utils.get_file(
            "style_predict.tflite",
            "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite",
        )
        style_transform_path = tf.keras.utils.get_file(
            "style_transform.tflite",
            "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite",
        )
        start_time = time.time()
        style_bottleneck = run_style_predict(style_image, style_predict_path)
        print(f"Style prediction time is {time.time() - start_time}")

        start_time = time.time()
        stylized_image = run_style_transform(
            style_bottleneck, content_image, style_transform_path
        )
        print(f"Style transfer time is {time.time() - start_time}")

        img = tf.keras.preprocessing.image.array_to_img(
            tf.squeeze(stylized_image).numpy(), data_format=None, scale=True, dtype=None
        )
    else:
        hub_handle = (
            "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
        )
        hub_module = hub.load(hub_handle)

        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]

        img = tf.keras.preprocessing.image.array_to_img(
            tf.squeeze(stylized_image).numpy(), data_format=None, scale=True, dtype=None
        )

    # write PNG in file-object
    if not send_image:
        img.save("output_frames/outputframe" + str(nframe) + ".jpg")
        # img.save("TEST.jpg")
    else:
        return img


def style_transfer_video_lite(n_frames, style_image_path=None):
    """
    Video style transfer for given number of frames using tf.lite version of the model.
    
    Args
        - n_frames (int): number of frames to style from test frames
    """
    style_predict_path = tf.keras.utils.get_file(
        "style_predict.tflite",
        "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite",
    )
    style_transform_path = tf.keras.utils.get_file(
        "style_transform.tflite",
        "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite",
    )
    start_time = time.time()
    style_image = preprocesses_style_image(style_image_path=style_image_path)
    style_bottleneck = run_style_predict(style_image, style_predict_path)
    print(f"Style prediction time is {time.time() - start_time}")
    print(style_bottleneck)

    frame_paths = []
    for i in range(n_frames):
        content_path = f"test_frames/testframe{i}.jpg"
        content_image = tf.convert_to_tensor(
            get_content_image_from_path(content_path), np.float32
        )
        print(f"on iteration {i}")
        interpreter = tf.lite.Interpreter(model_path=style_transform_path)

        # Set model input.
        input_details = interpreter.get_input_details()
        interpreter.allocate_tensors()
        start_time = time.time()
        # Set model inputs.
        interpreter.set_tensor(input_details[0]["index"], content_image)
        interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
        interpreter.invoke()

        # Transform content image.
        stylized_image = interpreter.tensor(
            interpreter.get_output_details()[0]["index"]
        )()
        print(f"Transfer time for frame {i} is {time.time() - start_time}")
        interpreter.reset_all_variables()

        img = tf.keras.preprocessing.image.array_to_img(
            tf.squeeze(stylized_image).numpy(), data_format=None, scale=True, dtype=None
        )

        img.save(f"output_frames/outputframe{i}.jpg")
        frame_paths.append(f"output_frames/outputframe{i}.jpg")
        # print(f"Frame {i} completed")
    return frame_paths


def style_transfer_video(n_frames, style_image_path=None):
    """
    Video style transfer for given number of frames using magenta model
    
    Args
        - n_frames (int): number of frames to style from test frames
    """
    style_image = preprocesses_style_image(style_image_path=style_image_path)

    start_time = time.time()
    hub_handle = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    hub_module = hub.load(hub_handle)
    print(f"Time to load hub module {time.time() - start_time}")
    frame_paths = []
    for i in range(n_frames):
        content_path = f"test_frames/testframe{i}.jpg"
        content_image = get_content_image_from_path(content_path)

        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]

        img = tf.keras.preprocessing.image.array_to_img(
            tf.squeeze(stylized_image).numpy(), data_format=None, scale=True, dtype=None
        )

        img.save(f"output_frames/outputframe{i}.jpg")
        frame_paths.append(f"output_frames/outputframe{i}.jpg")
        # print(f"Frame {i} completed")
    return frame_paths


def style_transfer_video_file(content_video_path, style_image_path, use_tflite):
    """
    Video style transfer for a given content video file fname

    Args
        - content_video_path (str): filepath of content video
        - style_image_path (str): filepath of style image
    """
    output_folder = "static"
    video_filename = "output.mp4"
    start_time = time.time()
    n_frames = slice_frames(content_video_path, "test_frames", "testframe")
    print(f"Time to slice up {time.time() - start_time} for {n_frames} frames")

    start_time = time.time()
    if use_tflite:
        frame_paths = style_transfer_video_lite(
            n_frames, style_image_path=style_image_path
        )
    else:
        frame_paths = style_transfer_video(n_frames, style_image_path=style_image_path)
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
