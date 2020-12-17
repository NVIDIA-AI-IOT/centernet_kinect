'''
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
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

from pipeline.setup import ModelSetup

"""
Utility functions for pipeline 
"""

def get_model():
    """
    The model weights are saved in CHECKPOINT_PATH specified in constants.py
    this functions loos into that directory and returns the path to the model.
    """
    model_path = const.CHECKPOINT_PATH
    models = glob(f"{model_path}/{const.LOSS}_{const.MODEL_NAME}_{const.DATA_LOADER}.pth")
    if not models:
        print(f"\nThere are not check points at:\
                \n{model_path}\
                \nPlease Train a model or change the directory on constants.py\n")
        exit -1
    return models[0]

def get_image(model_setup: ModelSetup, train=False):
    """
    Get a set of images and grounds truth from the dataset

    :param model_setup: ModelSetup
    :param train: bool, weather or not to chose from the training dataset
    """
    if train:
        idx = random.randint(0, model_setup.train_dataset.__len__())
        image, ground_truth = model_setup.train_dataset.dataset[idx]
    else:
        idx = random.randint(0, model_setup.valid_dataset.__len__())
        image, ground_truth = model_setup.valid_dataset.dataset[idx]
    return image.unsqueeze(0), ground_truth.unsqueeze(0)

def get_bboxes(yx_locations: torch.tensor, height: torch.tensor, width: torch.tensor,\
        offset_x: torch.tensor, offset_y: torch.tensor, stride=const.STRIDE, img_shape=const.IMG_SHAPE):
        """
        Create a list of bounding boxes [[xmin, ymin, xmax, ymax], ...]

        :param yx_locations: torch.tensor, X and Y locations in the heatmap has to be mutiplied by the stride to go back to original dims
        :param height: torch.tensor, The height of the bbox 
        :param width: torch.tensor, The width of the bbox
        :param offset_x: torch.tensor, The X offset value
        :param offst_y: torch.tensor, The Y offset value
        """
        yx_locations *= stride
        bboxes = []
        for i, yx_location in enumerate(yx_locations):
            y_center = yx_location[0].item() + offset_y[i].item()
            x_center = yx_location[1].item() + offset_x[i].item()
            h = height[i].item()
            w = width[i].item()

            x_min = max(0, x_center - w/2)
            y_min = max(0, y_center - h/2)

            bboxes.append([x_min, y_min, w, h])
        
        return bboxes

def find_prediction_mask(pred_heatmap: torch.tensor, window_size=11, threshold=0.3):
    """
    Find the mask of a giver heatmap, Have this in mind the follwoing heatmap might not have values as larg as
    1, and we need to fins the local maximas of the heatmap.

    :param pred_heatmap: torch.tensor, predicted heatmap by the model
    :param window_size: int, size of the maxPooling window
    :return: torch.tensor (mask of the heatmap)
    """
    pred_local_max = torch.max_pool2d(pred_heatmap[None, None, ...], kernel_size=window_size, stride=1, padding=window_size//2)
    return (pred_local_max == pred_heatmap) * (pred_heatmap > threshold)
