import os
import sys
import cv2
import json
import torch
import random
import numpy as np

from math import sqrt
from torch.utils.data import Dataset

# Adding Project Path
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(PROJ_PATH)

# Importing Project Libraries
import pipeline.constants as const

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# random.seed(42) # Answer to everything

def Transform(image: np.array, boxes: np.array, labels: np.array, train: bool):
    """
    Apply transformation.

    :param image: np.array(uint16), Depth image
    :param boxes: np.array(np.float), a list of N boxes of this image
    :param labels:  np.array(np.uint8), a lost of N labels of the boxes
    :param train: bool, if the item if for training of testing
    :return: Transformed image, transformed Boxes, transformed labels
    """
    new_image = image
    new_boxes = boxes
    new_labels = labels

    if train:
        if random.random() > 0.5:
            new_image = distort_image(new_image)

        if random.random() > 0.5:
            new_image = random_magnification(new_image, new_boxes)

        if random.random() > 0.5:
            new_image, new_boxes = zoom_out(new_image, new_boxes)

        if random.random() > 0.5:
            new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)

        if random.random() > 0.5:
            new_image, new_boxes = flip(new_image, new_boxes)

    new_image = new_image.astype(np.float32)

    return new_image, new_boxes, new_labels

def CreateHeatMap(img:np.array, bboxes:np.array, labels: np.array, img_shape=const.IMG_SHAPE,\
                num_classes=const.NUM_CLASSES, stride=const.STRIDE):
    """
    Create the HeatMap of the corresponding image with the given bounding boxes of the givern size img_shape
    
    :param img: np.array (uint16), the input depth image
    :param bboxes: np.array (float32), the bounding boxes of the input image
    :param labels:  np.array(np.uint8), a list of N labels of the boxes
    :param img_shape: tubple (w, h), the shape of the input image
    :param num_classes: int, number of classes in the dataset
    :param stride: int, the downsampling facor of the model
    """
    # Resize the original image to input dimentions
    input_image = cv2.resize(img, img_shape, interpolation=cv2.INTER_NEAREST)
    input_image = normalize(input_image)
    
    # Normalize the bounding boxes to output dimentions
    h = img.shape[0]
    w = img.shape[1]
    old_dims = np.expand_dims(np.array([w, h, w, h], dtype=np.float32), axis=0)
    
    output_shape = (img_shape[0]//const.STRIDE, img_shape[1]//const.STRIDE) # img_shape
    output_h = output_shape[0]
    output_w = output_shape[1]
    output_dims = np.expand_dims(np.array([output_w, output_h, output_w, output_h], dtype=np.float32), axis=0)
    
    new_bboxes = bboxes / old_dims * output_dims

    # Create an output tensor containing heatmap, box sizes, and offsets 
    output_tensor = np.zeros((num_classes+4, output_shape[0], output_shape[1]), dtype=np.float32)

    # Create a X,Y axis grid to compute the exponentials
    out_x = np.arange(output_shape[1]) + 0.5
    out_y = np.arange(output_shape[0]) + 0.5
    xy_grid = np.stack(np.meshgrid(out_x, out_y))

    for new_bbox, label in zip(new_bboxes, labels):
        x_min = new_bbox[0]
        y_min = new_bbox[1]
        x_max = new_bbox[2]
        y_max = new_bbox[3]

        center_x = min((x_min + x_max) / 2, output_h-1)
        center_y = min((y_min + y_max) / 2, output_w-1)
        width = abs(x_max - x_min)
        height = abs(y_max - y_min)

        sigma = sqrt(width**2 + height**2)

        heatmap = np.zeros((output_shape[0], output_shape[1]), dtype=np.float32)
        heatmap += (xy_grid[0] - int(center_x))**2
        heatmap += (xy_grid[1] - int(center_y))**2
        heatmap /= (sigma/10*const.STRIDE)
        heatmap *= -1
        heatmap = np.exp(heatmap)
        heatmap[heatmap < 0.02] = 0
        heatmap[int(center_y), int(center_x)] = 1

        # Heatmap
        output_tensor[label-1] = np.where(output_tensor[label-1] > heatmap, output_tensor[label-1], heatmap)
        # Size
        output_tensor[-4:-2, int(center_y), int(center_x)] = np.array([height*const.STRIDE, width*const.STRIDE])
        # offset
        output_tensor[-2:, int(center_y), int(center_x)] = np.array([center_y, center_x]) - np.floor(np.array([center_y, center_x]))
    
    input_image = np.expand_dims(input_image, 0)

    return input_image, output_tensor, new_bboxes # (Comment for test)



############################
###  Internal Functions  ###
############################

def distort_image(image: np.array):
    """
    Dirstort brightness, contrast

    :param img: np.array, Depth Image
    :return: np.array, Distorted Depth Image
    """
    def adjust_contrast(image: np.array):
        adjust_factor = random.uniform(0.1, 5)
        return np.clip(adjust_factor * image, 0, 65535).astype(np.uint16)
        # return np.clip(32768 + adjust_factor * image - adjust_factor * 32768, 0, 65535).astype(np.uint16)
    
    def adjust_brightness(image: np.array):
        adjust_factor = random.randint(-100, 900)
        return np.clip(image + adjust_factor , 0, 65535).astype(np.uint16)

    image = adjust_contrast(image)
    image = adjust_brightness(image)

    return image

def random_magnification(image: np.array, boxes: np.array):
    """
    Perform hand magnificatioon in an image by multiplying a random number in range [1, MAX_MAGNIFICATION]
    by the pixel values in the hand region

    Helps learning futher away values

    :param img: np.array, Depth Image (1, h, w)
    :param boxes: np.array, bounding boxes of the objects
    :return: expanded image, updated coordinates of bounding box
    """
    image = image.astype(np.float32)
    factor = random.uniform(0.4, const.MAX_MAGNIFICATION)
    for box in boxes:
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[2])
        ymax = int(box[3])
        image[ymin:ymax, xmin:xmax] *= factor
    
    return image

def zoom_out(image: np.array, boxes: np.array):
    """
    Perform zooming out of an image by placing the image in a larger canvas
    of filler values.
    filler will be the mean of the image

    Helps learning smaller values

    :param img: np.array, Depth Image (1, h, w)
    :param boxes: np.array, bounding boxes of the objects
    :return: expanded image, updated coordinates of bounding box
    """
    h = image.shape[0]
    w = image.shape[1]
    max_scale = const.MAX_ZOOM_OUT
    scale = random.uniform(1, max_scale)
    new_h = int(h*scale)
    new_w = int(w*scale)

    filler = image.mean()
    new_image = np.ones((new_h, new_w), dtype=np.float) * filler

    left = random.randint(0, new_w - w)
    right = left + w
    top = random.randint(0, new_h - h)
    bottom = top + h
    new_image[top:bottom, left:right] = image

    new_boxes = boxes + np.array([left, top, left, top], dtype=np.float32)

    return new_image, new_boxes
    

def random_crop(image: np.array, boxes: np.array, labels: np.array):
    """
    Performs a random crop in the manner stated in the paper.
    Helps detecting partial objects

    :param img: numpy Array, (1, h, w)
    :param boxes: numpy Array, bounding boxes of the objects
    :param labels: numpy Array, a lost of N labels of the boxes
    :return: cropped image, boxes, and the remaining labels in the imag
    """
    h = image.shape[0]
    w = image.shape[1]
    while True:
        # Randomly draw a value for min_overlap
        
        min_overlap = random.choice([0., .5, .7, .9, None])
        if min_overlap is None:
            return image, boxes, labels
        
        # Try 50 times for this choic of min_overlap
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimentions must be in range [0.3, 1] of the original image
            # Note - its [0.1, 1] in the paper
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = h*scale_h
            new_w = w*scale_w

            # Aspect ratio must be in range [0.5, 2]
            aspect_ratio = new_h / new_w
            if not (0.5 < aspect_ratio < 2):
                continue


            # Crop coordinates
            left = random.randint(0, int(w - new_w))
            right = int(left + new_w)
            top = random.randint(0, int(h - new_h))
            bottom =int( top + new_h)
            crop = np.expand_dims(np.array([left, top, right, bottom], dtype=np.float32), axis=0)

            # Calculate IoU for crop and bounding boxs
            overlap_crop = torch.FloatTensor(crop)
            overlap_boxes = torch.FloatTensor(boxes)
            overlap = find_jaccard_overlap(overlap_crop, overlap_boxes) # (1, n_objects)
            overlap = overlap.squeeze(0) # (n_objects)

            # If not a single bounding box satisfies the min overlap try again
            if overlap.max().item() < min_overlap:
                continue

            
            new_image = image[top:bottom, left:right] # (1, new_h, new_w)

            # Find center of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2. # (n_objects, 2)

            # Find Bounding Boxes whos center is in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) \
                            * (bb_centers[:, 1] > top) * (bb_centers[:, 1] < bottom)
            
            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that are not in the cropped image
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]

            # Calculate new bounding box coordinates
            # crop : (left, top, right, bottom)
    
            new_boxes[:, :2] = np.maximum(new_boxes[:, :2], crop[:, :2])
            new_boxes[:, :2] -= crop[:, :2]
            new_boxes[:, 2:] = np.minimum(new_boxes[:, 2:], crop[:, 2:])
            new_boxes[:, 2:] -= crop[:, :2]

            return new_image, new_boxes, new_labels


def flip(image: np.array, boxes: np.array):
    """
    Flip the image horizantally for better augmentation

    :param img: numpy array of Depth image
    :param boxes: numpy array, (n_objects, 4)
    :return: fliped image and bounding boxes.
    """
    h = image.shape[0]
    w = image.shape[1]
    random_rotate = random.choice([90, 180, 270])
    matrix = cv2.getRotationMatrix2D((h/2, w/2), random_rotate, 1)

    new_image = cv2.warpAffine(image, matrix, (h, w))

    new_boxes = np.ones( (boxes.shape[0]*2, 3) )
    new_boxes[:,:2] = boxes.reshape((boxes.shape[0]*2, 2))

    new_boxes = np.matmul(matrix, new_boxes.transpose())

    new_boxes = new_boxes.transpose()
    
    new_boxes = new_boxes.reshape(boxes.shape)

    for i, box in enumerate(new_boxes):
        xmin = min(box[0], box[2])
        ymin = min(box[1], box[3])
        xmax = max(box[0], box[2])
        ymax = max(box[1], box[3])
        new_boxes[i] = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

    return new_image, new_boxes

def normalize(image: np.array):
    """
    Resize image to (300, 300)

    :param image: numpy array
    :return: normalized image
    """
    mean = np.mean(image)
    std = image.std()
    if std==0:
        std = 1 
    new_image = (image - mean) / std
    return new_image

def find_intersection(set_1: torch.tensor, set_2: torch.tensor):
    """
    Find the intersection of every box combination betweeen 2 sets of boxes that are in boundary coordinates.

    :param set_1: set_1 (n1, 4)
    :param set_2: set 2 (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the set 2 (n1, n2)
    """
    DEVICE = set_1.device
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1).to(DEVICE), set_2[:, :2].unsqueeze(0).to(DEVICE)) # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1).to(DEVICE), set_2[:, 2:].unsqueeze(0).to(DEVICE)) # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0) # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1] # (n1, n2)

def find_jaccard_overlap(set_1: torch.tensor, set_2: torch.tensor):
    """
    Find IoU of every box combination in between the 2 sets (boxes in boundary coordinates)

    :param set_1: set 1 (n1, 4)
    :param set2: set 2 (n2, 4)
    :return: Jaccard overlap of each of the boxes in the set 1 with respect to set 2 (n1, n2)
    """
    DEVICE = set_1.device

    intersection = find_intersection(set_1, set_2)

    area_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1]) # (n1)
    area_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1]) # (n1)

    union = area_set_1.unsqueeze(1).to(DEVICE) + area_set_2.unsqueeze(0).to(DEVICE) - intersection # (n1, n2)

    return intersection / union


###################
###   Testing   ###
###################
"""
import torch 
import os 
import torchvision.transforms.functional as FT
import matplotlib.patches as patches
import matplotlib.pyplot as plt 
from PIL import Image, ImageOps 
import json 
import cv2 
import numpy as np

with open("/home/analog/Desktop/NVIDIA/SSD/preprocess/data/train.json", "r") as jf: 
    load_dict = json.load(jf)

img_annotation = load_dict[100]

img_num = img_annotation["img_path"].split("/")[-1].split(".")[0]  
base_path = "/home/analog/Desktop/NVIDIA/DataCollection/dataset/data/"  
img_path = os.path.join(base_path, "ir_image/"+img_num+".png") 

boxes = np.array(img_annotation["boxes"], dtype=np.float32) # Boxes need to be casted into Numpy Float Array 
labels = np.array(img_annotation["labels"], dtype=np.long) # Labels need to be casted into torch Long tensor 
image = cv2.imread(img_path, cv2.COLOR_BGR2GRAY).astype(np.uint16) # 16 bit unsigned integer values for an Depth Image (H, W)    


new_image, new_boxes, new_labels = Transform(image, boxes, labels, True)
new_image, heat_map, new_boxes = CreateHeatMap(new_image, new_boxes, new_labels)

fig = plt.figure(figsize=(6, 8))
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)


for i in range(new_boxes.shape[0]):
    box_locs = new_boxes[i].tolist()
    x, y = box_locs[0], box_locs[1]
    width, height = abs(box_locs[0] - box_locs[2]), abs(box_locs[1] - box_locs[3])
    rect = patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='r',facecolor='none')
    ax2.add_patch(rect)


out_img = new_image[0]

ax1.imshow(out_img, interpolation='nearest', cmap ='gray')

out_img = heat_map[0]
ax2.imshow(out_img, interpolation='nearest', cmap ='gray')

plt.show()
"""