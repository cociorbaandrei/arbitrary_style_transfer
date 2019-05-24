# Utility

from os import listdir, mkdir, sep
from os.path import join, exists, splitext

import cv2
import matplotlib.pyplot as plt
import numpy as np


def list_images(directory):
    images = []
    for file in listdir(directory):
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
    return images


def get_train_images(paths, resize_len=512, crop_height=256, crop_width=256):
    images = []
    for path in paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        if height < width:
            new_height = resize_len
            new_width = int(width * new_height / height)
        else:
            new_width = resize_len
            new_height = int(height * new_width / width)

        image = cv2.resize(image, (new_width, new_height))

        # crop the image
        start_h = np.random.choice(new_height - crop_height + 1)
        start_w = np.random.choice(new_width - crop_width + 1)
        image = image[start_h:(start_h + crop_height), start_w:(start_w + crop_width), :]

        images.append(image)

    images = np.stack(images, axis=0)

    return images


def get_images(paths, height=None, width=None):
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if height is not None and width is not None:
            image = cv2.resize(image, (width, height))

        images.append(image)

    images = np.stack(images, axis=0)

    return images


def save_images(datas, contents_path, styles_path, save_dir, suffix=None):
    assert (len(datas) == len(contents_path) * len(styles_path))

    if not exists(save_dir):
        mkdir(save_dir)

    if suffix is None:
        suffix = ''

    data_idx = 0
    for content_path in contents_path:
        for style_path in styles_path:
            data = datas[data_idx]
            data_idx += 1

            content_path_name, content_ext = splitext(content_path)
            style_path_name, style_ext = splitext(style_path)

            content_name = content_path_name.split(sep)[-1]
            style_name = style_path_name.split(sep)[-1]

            save_path = join(save_dir, '%s-%s%s%s' %
                             (content_name, style_name, suffix, content_ext))
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, data)


def plot_side_by_side(imgs, titles=None):
    fig = plt.figure(figsize=(20, 20))
    columns = len(imgs)
    rows = 1
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i, title=titles[i - 1] if titles is not None else None)
        try:
            img = imgs[i - 1]
        except IndexError:
            img = imgs[i - 1].squeeze(axis=2)
        if len(img.shape) == 3:
            plt.imshow(img[:, :, [2, 1, 0]], cmap='Greys_r')
        else:
            plt.imshow(img, cmap='Greys_r')

    plt.show()
