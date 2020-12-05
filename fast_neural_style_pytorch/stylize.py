import os
import time

import cv2
import torch
from torchvision import transforms

import fast_neural_style_pytorch.experimental as experimental
import fast_neural_style_pytorch.transformer as transformer
import fast_neural_style_pytorch.utils as utils

STYLE_TRANSFORM_PATH = "fast_neural_style_pytorch/transforms/udnie.pth"
PRESERVE_COLOR = False


def stylize(
    content_image_path=None,
    style_path="fast_neural_style_pytorch/transforms/starry.pth",
):
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Transformer Network
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(style_path, map_location=torch.device("cpu")))
    net = net.to(device)

    with torch.no_grad():
        while 1:
            torch.cuda.empty_cache()
            print("Stylize Image~ Press Ctrl+C and Enter to close the program")
            if content_image_path == None:
                content_image_path = input("Enter the image path: ")
            content_image = utils.load_image(content_image_path)
            starttime = time.time()
            content_tensor = utils.itot(content_image).to(device)
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())
            if PRESERVE_COLOR:
                generated_image = utils.transfer_color(content_image, generated_image)
            print("Transfer Time: {}".format(time.time() - starttime))
            # utils.show(generated_image)
            # utils.saveimg(generated_image, "helloworld.jpg")
            return generated_image


def stylize_folder_single(style_path, content_folder, save_folder):
    """
    Reads frames/pictures as follows:

    content_folder
        pic1.ext
        pic2.ext
        pic3.ext
        ...

    and saves as the styled images in save_folder as follow:

    save_folder
        pic1.ext
        pic2.ext
        pic3.ext
        ...
    """
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Transformer Network
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(style_path, map_location=torch.device("cpu")))
    net = net.to(device)

    # Stylize every frame
    images = [img for img in os.listdir(content_folder) if img.endswith(".jpg")]
    with torch.no_grad():
        for image_name in images:
            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()

            # Load content image
            content_image = utils.load_image(content_folder + image_name)
            content_tensor = utils.itot(content_image).to(device)

            # Generate image
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())
            if PRESERVE_COLOR:
                generated_image = utils.transfer_color(content_image, generated_image)
            # Save image
            utils.saveimg(generated_image, save_folder + image_name)


def stylize_folder(
    style_path,
    folder_containing_the_content_folder,
    save_folder,
    batch_size=1,
    prune_level=1.0,
):
    """Stylizes images in a folder by batch
    If the images  are of different dimensions, use transform.resize() or use a batch size of 1
    IMPORTANT: Put content_folder inside another folder folder_containing_the_content_folder

    folder_containing_the_content_folder
        content_folder
            pic1.ext
            pic2.ext
            pic3.ext
            ...

    and saves as the styled images in save_folder as follow:

    save_folder
        pic1.ext
        pic2.ext
        pic3.ext
        ...
    """
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Image loader
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))]
    )
    image_dataset = utils.ImageFolderWithPaths(
        folder_containing_the_content_folder, transform=transform
    )
    image_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size)

    # Load Transformer Network
    net = transformer.TransformerNetwork()
    # net = experimental.TransformerResNextNetwork_Pruned(alpha=prune_level)
    print(f"style path is {style_path}")
    net.load_state_dict(torch.load(style_path, map_location=torch.device("cpu")))
    net = net.to(device)

    # Stylize batches of images
    with torch.no_grad():
        for content_batch, _, path in image_loader:
            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()

            # Generate image
            generated_tensor = net(content_batch.to(device)).detach()

            # Save images
            for i in range(len(path)):
                generated_image = utils.ttoi(generated_tensor[i])
                if PRESERVE_COLOR:
                    generated_image = utils.transfer_color(
                        content_image, generated_image
                    )
                image_name = os.path.basename(path[i])
                utils.saveimg(generated_image, save_folder + image_name)


if __name__ == "__main__":
    stylize()
