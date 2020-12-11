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
from dataLoader.fused_utils import Transform, CreateHeatMap

class FusedProjDataLoader(Dataset):
    """
    Pytorch data loader

    Create HeatMaps for each input depth image from the given bouning box
    """
    def __init__(self, dataset_path=const.DATASET_PATH, json_annotation_path=const.JSON_ANNOTATION_PATH, train=True,\
                    ir_img_dir_path=const.IR_IMG_DIR_PATH, depth_img_dir_path=const.DEPTH_IMG_DIR_PATH):
        """
        Class Constructor

        :param dataset_path: str, full path to dataset location
        :param json_annotation_path: str, full path to json annotation directory
        """
        self.dataset_path = dataset_path
        self.ir_img_dir_path = ir_img_dir_path
        self.depth_img_dir_path = depth_img_dir_path
        self.train = train

        self.depth_path = os.path.join(dataset_path, "data/depth_image")
        self.ir_path = os.path.join(dataset_path, "data/ir_image")

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
        :return: normalize resized (3x320x320) image [depth, ir, (ir+depth)/2] (torch.tensor), [heatmaps, sizes, offsets] (1+num_classses, 80, 80) (torch.tensor)
        """
        img_annotation = self.img_annotations[idx]

        img_num = img_annotation["img_path"].split("/")[-1]
        depth_img_path = os.path.join(self.depth_img_dir_path, f"{img_num}")
        ir_img_path = os.path.join(self.ir_img_dir_path, f"{img_num}")

        bboxes = np.array(img_annotation["boxes"], dtype=np.float32) # Boxes need to be casted into Numpy Float Array
        labels = np.array(img_annotation["labels"], dtype=np.long) # Labels need to be casted into torch Long tensor
        depth_image = cv2.imread(depth_img_path, cv2.COLOR_BGR2GRAY).astype(np.uint16) # 16 bit unsigned integer values for an IR Image (H, W)
        ir_image = cv2.imread(ir_img_path, cv2.COLOR_BGR2GRAY).astype(np.uint16) # 16 bit unsigned integer values for an IR Image (H, W)

        new_depth_image, new_ir_image, new_boxes, new_labels = Transform(depth_image, ir_image, bboxes, labels, self.train)
        new_depth_image, new_ir_image, output_tensor, new_boxes = CreateHeatMap(new_depth_image, new_ir_image, new_boxes, new_labels)

        output_tensor = torch.from_numpy(output_tensor).type(torch.float32)
        
        new_image_c1 = torch.from_numpy(new_depth_image).type(torch.float32)
        new_image_c2 = torch.from_numpy(new_ir_image).type(torch.float32)
        new_image_c3 = (new_image_c1 + new_image_c2) / 2

        new_image = torch.cat((new_image_c1, new_image_c2, new_image_c3), 0) 

        return new_image, output_tensor