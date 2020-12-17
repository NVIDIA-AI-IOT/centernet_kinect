import os
import sys
import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from glob import glob
from pyk4a import PyK4A
from PIL import Image, ImageOps
# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

# PROJ LIBRARY
import pipeline.constants as const

from pipeline.utils import *
from pipeline.setup import ModelSetup
from model.run import Run_Inference


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    """
    Argument parser function for main.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load',
                        type=str,
                        default=None,
                        help="Set the full path to your trained model")

    args = parser.parse_args()
    return args


def normalize(image: np.array, img_shape=const.IMG_SHAPE):
    """
    Resize image to (300, 300)

    :param image: numpy array
    :return: normalized image Casted to torch
    """
    image = cv2.resize(image, img_shape, interpolation=cv2.INTER_NEAREST)
    mean = np.mean(image)
    std = image.std()
    if std==0:
        std = 1 
    new_image = (image - mean) / std

    # Cast to pytorch and expand dimentions for the model forward pass    
    new_image = torch.from_numpy(new_image).type(torch.float32)

    new_image = new_image.unsqueeze(0)

    return new_image

def input_image(depth_image: np.array, ir_image: np.array, img_shape=const.IMG_SHAPE, input_type=const.DATA_LOADER):
    """
    Transform the images into the input images

    :param depth_img: np.array (uint16), depth image
    :param ir_image: np.array (uint16), depth image
    :param img_shape: tuple (h, w), image size
    :param input_type: str, with input type (Fused, Depth) is the model using
    :return: 
    """
    def depth_input(depth_image: np.array, **kwargs):
        c, h, w = depth_image.size()
        depth_image = depth_image[0:1,:,:] # depth
        new_image = depth_image.expand(1, 3, h, w)
        return new_image

    def fused_input(depth_image: np.array, ir_image:np.array):
        c, h, w = depth_image.size()
        new_image_c1 = depth_image.type(torch.float32)
        new_image_c2 = ir_image.type(torch.float32)
        new_image_c3 = (new_image_c1 + new_image_c2) / 2
        new_image = torch.cat((new_image_c1, new_image_c2, new_image_c3), 0)
        new_image = new_image.expand(1, 3, h, w)
        return new_image

    input_switcher = {
        "depth": depth_input,
        "fused": fused_input,
    }
    depth_image = normalize(depth_image)
    ir_image = normalize(ir_image)
    
    return input_switcher[input_type](depth_image=depth_image, ir_image=ir_image)

def get_depth(depth_img: np.array, bbox: list, img_shape=const.IMG_SHAPE):
    """
    Given the input depth image and the bounding box coordinates return the median depth of the region

    :param depth_img: np.array (uint16), depth image
    :param bbox: list, [x, y, w, h]
    :param img_shape: tuple (h, w), image size
    :retun: (float) median depth of the bbox  in the image
    """
    x, y = int(bbox[0]), int(bbox[1])
    width, height = int(bbox[2]), int(bbox[3])

    depth_img = cv2.resize(depth_img, img_shape, interpolation=cv2.INTER_NEAREST)

    return np.median(depth_img[y:y+height, x:x+width])

def get_mask(depth_img: np.array, bboxs: list, img_shape=const.IMG_SHAPE, depth_thresh=10):
    """
    Given the input depth image and the bounding box coordinates return the masks of the hand

    :param depth_img: np.array (uint16), depth image
    :param bboxs: list, [x, y, w, h]
    :param img_shape: tuple (h, w), image size
    :param depth_thresh: threshold to mask the pixels
    :retun: np.array, mask of the hands
    """
    depth_img = cv2.resize(depth_img, img_shape, interpolation=cv2.INTER_NEAREST)
    mask = np.zeros(depth_img.shape, dtype=np.uint8)

    for bbox in bboxs:    
        x, y = int(bbox[0]), int(bbox[1])
        width, height = int(bbox[2]), int(bbox[3])
        median_depth = np.median(depth_img[y:y+height, x:x+width])

        mask[y:y+height, x:x+width] = depth_img[y:y+height, x:x+width]
        mask[np.where(mask[y:y+height, x:x+width] >= median_depth + depth_thresh)] = 0 
        mask[np.where(mask[y:y+height, x:x+width] <= median_depth - depth_thresh)] = 0 

    mask[np.where(mask > 0.1)] = 1
    return mask

def run_camera_inferance(model_setup: ModelSetup, iterations=1000):
    """
    Run the model for N number of frames

    :param model_setup: ModelSetup
    :param iterations: the total number of frames to run the model
    """
    fig = plt.figure(figsize=(8, 10))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    # ax3 = fig.add_subplot(3, 2, 1)
    ax4 = fig.add_subplot(3, 1, 3)

    # Load camera with default config
    k4a = PyK4A()
    k4a.start()

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    for i in range(iterations):
        capture = k4a.get_capture()
        ir_img = capture.ir
        depth_img = capture.depth

        # ir_img[ir_img > 10000] = ir_img.mean()

        # Take this out just to test with PIL
        ir_img = Image.fromarray(ir_img)
        ir_img = ImageOps.grayscale(ir_img)
        ir_img = np.array(ir_img)
        # End here

        w, h = ir_img.shape[1], ir_img.shape[0]
        transformed_image = input_image(depth_img, ir_img, const.IMG_SHAPE, model_setup.input_format)


        # Run Infrence
        prediction = Run_Inference(model_setup, transformed_image)

        pred_heatmap = prediction[0][0:model_setup.num_classes].max(0)[0].float()
        pred_mask = find_prediction_mask(pred_heatmap)[0][0]
        pred_yx_locations = torch.nonzero(pred_mask)

        pred_height = prediction[0][-4][pred_mask]
        pred_width = prediction[0][-3][pred_mask]

        pred_offset_y = prediction[0][-2][pred_mask]
        pred_offset_x = prediction[0][-1][pred_mask]

        pred_bboxes = get_bboxes(pred_yx_locations, pred_height, pred_width, pred_offset_x, pred_offset_y)    

        for pred_box in pred_bboxes:
            x, y = pred_box[0], pred_box[1]
            width, height = pred_box[2], pred_box[3]
            rect = patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='g',facecolor='none')
            ax1.add_patch(rect)
            ax1.text(x, y, str(get_depth(depth_img, pred_box)), fontsize=14, bbox=props)


        ir_img = cv2.resize(ir_img, const.IMG_SHAPE, interpolation=cv2.INTER_NEAREST)
        ax1.imshow(ir_img, interpolation='nearest', cmap ='gray')

        ax2.imshow(pred_heatmap.cpu().numpy(), interpolation='nearest', cmap ='gray')

        mask = get_mask(depth_img, pred_bboxes)
        # ax3.imshow(mask)

        ax4.imshow(pred_mask.float().cpu().numpy(), interpolation='nearest', cmap ='gray')

        plt.draw()
        plt.pause(0.001)

        ax1.clear()
        ax2.clear()
        # ax3.clear()
        ax4.clear()

        # del rect, capture, prediction

def main():
    args = parse_arguments()
    if args.load:
        model_setup = ModelSetup(load=args.load, infer=True)
    else:
        model_setup = ModelSetup(load=get_model(), infer=True)
    
    run_camera_inferance(model_setup)

if __name__ == "__main__":
    main()