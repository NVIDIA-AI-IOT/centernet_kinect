'''
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import cv2
import json
import torch
import numpy as np

from torch.utils.data import Dataset

# Adding Project Path
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(PROJ_PATH)

# Importing Project Libraries
import pipeline.constants as const
from dataLoader.utils import Transform, CreateHeatMap

class ProjDataLoader(Dataset):
    """
    Pytorch data loader

    Create HeatMaps for each input depth image from the given bouning box
    """
    def __init__(self, json_annotation_path=const.JSON_ANNOTATION_PATH, train=True,\
                    ir_img_dir_path=const.IR_IMG_DIR_PATH, depth_img_dir_path=const.DEPTH_IMG_DIR_PATH):
        """
        Class Constructor

        :param dataset_path: str, full path to dataset location
        :param json_annotation_path: str, full path to json annotation directory
        """
        self.ir_img_dir_path = ir_img_dir_path
        self.depth_img_dir_path = depth_img_dir_path
        self.train = train

        if train:
            self.annotaion_path = os.path.join(json_annotation_path, "train.json")
        else:
            self.annotaion_path = os.path.join(json_annotation_path, "val.json")

        with open(self.annotaion_path, 'r') as fp:
            self.img_annotations = json.load(fp)

        self.length = len(self.img_annotations)
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        :param idx: int, index of the item to retreave
        :return: normalize resized (3x320x320) image [depth, depth, depth] (torch.tensor), [heatmaps, sizes, offsets] (1+num_classses, 80, 80) (torch.tensor)
        """
        img_annotation = self.img_annotations[idx]

        img_num = img_annotation["img_path"].split("/")[-1]
        img_path = os.path.join(self.depth_img_dir_path, f"{img_num}")

        bboxes = np.array(img_annotation["boxes"], dtype=np.float32) # Boxes need to be casted into Numpy Float Array
        labels = np.array(img_annotation["labels"], dtype=np.long) # Labels need to be casted into torch Long tensor
        image = cv2.imread(img_path, cv2.COLOR_BGR2GRAY).astype(np.uint16) # 16 bit unsigned integer values for an IR Image (H, W)

        new_image, new_boxes, new_labels = Transform(image, bboxes, labels, self.train)
        new_image, output_tensor, new_boxes = CreateHeatMap(new_image, new_boxes, new_labels)

        output_tensor =  torch.from_numpy(output_tensor).type(torch.float32)
        new_image = torch.from_numpy(new_image).type(torch.float32)

        c, h, w = new_image.size()

        new_image = new_image[0:1,:,:] # depth
        new_image = new_image.expand(3, h, w)


        return new_image, output_tensor

