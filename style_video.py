import cv2
from PIL import Image
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_hub as hub
import io
import functools

def slice_frames(video_file):
    """ video -> images """
    cap = cv2.VideoCapture(video_file)

    idx = 0
    framecount = 0
    frame_skip = 10
    while (cap.isOpened()):
        ret, frame = cap.read()
        if (ret):
            if (idx == frame_skip):
                filename = "test_frames/testframe" + str(framecount) + ".jpg"
                cv2.imwrite(filename, frame)
                framecount += 1
                idx = 0
            else:
                idx += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    return idx

def combine_frames():
    """images -> frames"""
    image_folder = 'output_frames'
    video_filename = 'output.mp4'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_filename, 0, 5, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
    """Loads and preprocesses images."""
    # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
    img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
    if img.max() > 1.0:
        img = img / 255.
    if len(img.shape) == 3:
        img = tf.stack([img, img, img], axis=-1)
    img = crop_center(img)
    img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
    return img

def preprocesses_style_image(style_image_url=None):
    if not style_image_url:
        style_image_url = "https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg"
        style_image_path = tf.keras.utils.get_file(os.path.basename(style_image_url)[-128:], style_image_url)

    style_image = load_image(style_image_path, style_img_size)
    style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')

    return style_image

def get_style_transfer(content_image_path, nframe, style_image):
    #style_image_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
    output_image_size = 384
    content_img_size = (output_image_size, output_image_size)
    style_img_size = (256, 256)

    content_image = load_image(content_image_path, content_img_size)
    # show_n([content_image, style_image], ['Content image', 'Style image'])

    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)

    outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
    stylized_image = outputs[0]

    img = tf.keras.preprocessing.image.array_to_img(
        tf.squeeze(stylized_image).numpy(), data_format=None, scale=True, dtype=None)
    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save("output_frames/outputframe" + str(nframe) + ".jpg")

def style_transfer_video(n_frames):
    style_image = preprocesses_style_image()
    for i in range(n_frames):
        content_url = "test_frames/testframe" + str(i) + ".jpg"
        get_style_transfer(content_url, i, style_image)

def style_transfer_video_file(fname):
    n_frames = slice_frames(fname)
    style_transfer_video(n_frames)
    combine_frames()


if __name__ == "__main__":
    #style_transfer_video(25)
    combine_frames()
