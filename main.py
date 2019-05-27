# Demo - train the style transfer network & use it to generate an image

from __future__ import print_function

from itertools import product

from infer import stylize
from utils import list_images

# for training
ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'

STYLE_WEIGHTS = [0.70]
MODEL_SAVE_PATHS = [
    'models/style_weight_2e0.ckpt',
]

# for inferring (stylize)
INFERRING_CONTENT_DIR = 'images/content'
INFERRING_STYLE_DIR = 'images/style'
OUTPUTS_DIR = 'outputs'


def main():
    content_imgs_path = list_images(INFERRING_CONTENT_DIR)
    style_imgs_path = list_images(INFERRING_STYLE_DIR)

    for style_weight, model_save_path in product(STYLE_WEIGHTS, MODEL_SAVE_PATHS):
        print('\n>>> Begin to stylize images with style weight: %.2f\n' % style_weight)

        stylize(content_imgs_path, style_imgs_path, OUTPUTS_DIR,
                ENCODER_WEIGHTS_PATH, model_save_path,
                suffix='-' + str(style_weight),
                alpha=style_weight)

    print('\n>>> Successfully! Done all stylizing...\n')


if __name__ == '__main__':
    main()
