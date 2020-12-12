import os
import sys
import torch
import random
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from glob import glob

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

# PROJ LIBRARY
import pipeline.constants as const

from pipeline.utils import *
from pipeline.setup import ModelSetup
from model.run import Run_Inference

# random.seed(42)

def parse_arguments():
    """
    Argument parser function for main.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load',
                        type=str,
                        default=None,
                        help="Set the full path to your trained model")

    parser.add_argument('-t', '--train',
                        type=bool,
                        default=False,
                        help="Set to True to use training set for visulization")

    args = parser.parse_args()
    return args

def draw_images(model_setup: ModelSetup, image: torch.tensor, ground_truth: torch.tensor, prediction: torch.tensor):
    """
    Draw the image with the corresponding bounding boxes and heatmaps

    :param model_setup: ModelSetup
    :param image: torch.tensor, the normalized input image to the model [N, C, H, W] (1, 1, 320, 320)
    :param ground_truth: torch.tensor, the corresponding ground truth of the input image (1, 1, 160, 160)
    :param prediction: torch.tensor, the corresponding prediction of the model (1, 1, 160, 160)
    """
    image = image.numpy()[0][0]

    gt_heatmap = ground_truth[0][0:model_setup.num_classes].max(0)[0].float()
    gt_mask = ground_truth[0][0:model_setup.num_classes].max(0)[0].eq(1)
    gt_yx_locations = torch.nonzero(gt_mask)

    gt_height = ground_truth[0][-4][gt_mask]
    gt_width = ground_truth[0][-3][gt_mask]

    gt_offset_y = ground_truth[0][-2][gt_mask]
    gt_offset_x = ground_truth[0][-1][gt_mask]

    gt_bboxes = get_bboxes(gt_yx_locations, gt_height, gt_width, gt_offset_x, gt_offset_y)


    pred_heatmap = prediction[0][0:model_setup.num_classes].max(0)[0].float()
    pred_mask = find_prediction_mask(pred_heatmap)[0][0]
    pred_yx_locations = torch.nonzero(pred_mask)

    pred_height = prediction[0][-4][pred_mask]
    pred_width = prediction[0][-3][pred_mask]

    pred_offset_y = prediction[0][-2][pred_mask]
    pred_offset_x = prediction[0][-1][pred_mask]

    pred_bboxes = get_bboxes(pred_yx_locations, pred_height, pred_width, pred_offset_x, pred_offset_y)    

    fig = plt.figure(figsize=(6, 11))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    for gt_box in gt_bboxes:
        x, y = gt_box[0], gt_box[1]
        width, height = gt_box[2], gt_box[3]
        rect = patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='g',facecolor='none')
        ax1.add_patch(rect)

    for pred_box in pred_bboxes:
        x, y = pred_box[0], pred_box[1]
        width, height = pred_box[2], pred_box[3]
        rect = patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='r',facecolor='none')
        ax1.add_patch(rect)


    ax1_img = image
    ax1.imshow(ax1_img, interpolation='nearest', cmap ='gray')
    ax1.set_title('IR Image')

    ax2_img = gt_heatmap.numpy()
    ax2.imshow(ax2_img, interpolation='nearest', cmap ='gray')
    ax2.set_title('Ground Truth Mask')

    ax3_img = pred_heatmap.cpu().numpy()
    ax3.imshow(ax3_img, interpolation='nearest', cmap ='gray')
    ax3.set_title('Prediction Mask')

    plt.show()

def main():
    args = parse_arguments()
    if args.load:
        model_setup = ModelSetup(load=args.load, infer=True)
    else:
        model_setup = ModelSetup(load=get_model(), infer=True)

    image, ground_truth = get_image(model_setup, args.train)

    predictions = Run_Inference(model_setup, image)

    draw_images(model_setup, image, ground_truth, predictions)
    

if __name__ == "__main__":
    main()