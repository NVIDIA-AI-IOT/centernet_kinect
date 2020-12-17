'''
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import json
import argparse
import numpy as np
import xml.etree.ElementTree as ET

# Adding Project Path
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(PROJ_PATH)

# Importing Project Libraries
import pipeline.constants as const

JSON_ANNOTATION_PATH = const.JSON_ANNOTATION_PATH
np.random.seed(42)


"""
the label map for our dataset only contains back ground and hand
"""
label_map = {
    "background" : 0,
    "hand" : 1,
}

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', "--annotation_path",\
                        default=const.DATASET_ANNOTATION_PATH)
    args = parser.parse_args()
    return args

def get_bounding_box(bb_obj):
    xmin = int(bb_obj.find('xmin').text) - 1
    ymin = int(bb_obj.find('ymin').text) - 1
    xmax = int(bb_obj.find('xmax').text) - 1
    ymax = int(bb_obj.find('ymax').text) - 1

    return [xmin, ymin, xmax, ymax]

def get_img_size(size_obj):
    width = int(size_obj.find("width").text)
    height = int(size_obj.find("height").text)
    channel = int(size_obj.find("depth").text)

    return [channel, height, width]

def pars_xml_file(xml_file):
    file_tree = ET.parse(xml_file)
    root = file_tree.getroot()

    ann_list = list()
    boxes = list()
    labels = list()
    for ann_obj in root.iter("annotation"):   
        path = ann_obj.find("path").text
        chw = get_img_size(ann_obj.find("size"))
        
        for obj in ann_obj.iter("object"):
            label = obj.find("name").text.lower().strip()
            labels.append(label_map[label])

            bbox = get_bounding_box(obj.find("bndbox"))
            boxes.append(bbox)

        dic = {'img_path': path, 'chw': chw, 'boxes': boxes, 'labels': labels}
        ann_list.append(dic)

    return ann_list

def choose_random_split(lis, split=10):
    split_size = int(split * len(lis) / 100)
    zeros = np.zeros(len(lis))
    zeros[np.random.choice(len(lis), size=split_size, replace=False)] = 1
    
    train = list()
    val = list()

    for idx, elem in enumerate(zeros):
        if elem == 0:
            train.append(lis[idx])
        else:
            val.append(lis[idx])

    return train, val

def main():
    args = parse_argument()
    annotations = list()
    for ann in os.listdir(args.annotation_path):
        if "xml" in ann.lower():
            ann_list = pars_xml_file(os.path.join(args.annotation_path, ann))
            for elem in ann_list:
                annotations.append(elem)
    
    train, val = choose_random_split(annotations, split=10)
    train_path = os.path.join(const.JSON_ANNOTATION_PATH, "train.json") 
    with open(train_path, "w") as j:
        json.dump(train, j)
    print(f"train data saved at: \n {train_path}")

    val_path = os.path.join(const.JSON_ANNOTATION_PATH, "val.json") 
    with open(val_path, "w") as j:
        json.dump(val, j)
    print(f"validation data saved at: \n {val_path}")

    label_map_path = os.path.join(const.JSON_ANNOTATION_PATH, "label_map.json") 
    with open(label_map_path, "w") as j:
        json.dump(label_map, j)
    print(f"label_map data saved at: \n {label_map_path}")



if __name__ == "__main__":
    main()