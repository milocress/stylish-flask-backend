import video
import click
from stylize import stylize_folder
import time

CONTENT_FRAME_PATH = "frames/content_folder/"
STYLE_FRAME_SAVE_PATH = "style_frames/"
STYLE_VIDEO_NAME = "pruned10.mp4"
STYLE_PATH = "transforms/mosaic_TransformerResNextNetwork_Pruned03.pth"
FRAME_SAVE_PATH = "frames/"
ORIGINAL_VIDEO_PATH = "cat.mp4"

@click.group()
def main():
	pass


@main.command()
@click.option("--style_path", default=STYLE_PATH)
@click.option("--frame_save_path", default=FRAME_SAVE_PATH)
@click.option("--style_frame_save_path", default=STYLE_FRAME_SAVE_PATH)
@click.option("--batch_size", default=20)
def stylize(style_path, frame_save_path, style_frame_save_path, batch_size):
    start_time = time.time()
    prune_level = 0
    if "10" in style_path:
        prune_level = 1.0
    elif "03" in style_path:
        prune_level = 0.3
    stylize_folder(style_path, frame_save_path, style_frame_save_path, batch_size=batch_size, prune_level=prune_level)
    print("Transfer time: {}".format(time.time() - start_time))

    start_time = time.time()
    H, W, fps = video.getInfo(ORIGINAL_VIDEO_PATH)
    video.makeVideo(STYLE_FRAME_SAVE_PATH, STYLE_VIDEO_NAME, fps, int(H), int(W))
    print("Collation time: {}".format(time.time() - start_time))


if __name__ == "__main__":
	main()
	# video_path = "cat.mp4"
	# H, W, fps = video.getInfo(video_path)
	# video.makeVideo(STYLE_FRAME_SAVE_PATH, STYLE_VIDEO_NAME, fps, int(H), int(W))
