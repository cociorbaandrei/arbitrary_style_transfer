import cv2
import os

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


def resize(img, shorter_size):
    h, w, _ = img.shape

    if h >= w:
        new_w = shorter_size
        new_h = shorter_size * h / w
    else:
        new_h = shorter_size
        new_w = shorter_size * w / h

    resized_img = cv2.resize(img, (int(new_w), int(new_h)))

    return resized_img


for im_name in os.listdir(INFERRING_STYLE_DIR):
    if im_name[-4:] in [".jpg", ".png"]:
        cont = cv2.imread(os.path.join(INFERRING_STYLE_DIR, im_name))
        resized = resize(cont, shorter_size=400)
        #         plot_side_by_side([cont, resized])
        cv2.imwrite(os.path.join(INFERRING_STYLE_DIR, im_name), resized)
