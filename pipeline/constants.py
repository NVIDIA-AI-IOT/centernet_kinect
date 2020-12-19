'''
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys

# Adding Project Path
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
PROJ_PATH = os.path.join(DIR_PATH, os.path.pardir)

# Setup the json annotation file paths
JSON_ANNOTATION_PATH = ""
JSON_ANNOTATION_PATH = os.path.join(PROJ_PATH, "annotation_json") if not JSON_ANNOTATION_PATH else JSON_ANNOTATION_PATH # Default path is not setup

# Seting up the dataset paths
SAVE_DATASET_PATH = ""
SAVE_DATASET_PATH = os.path.join(JSON_ANNOTATION_PATH, "data") if not SAVE_DATASET_PATH else SAVE_DATASET_PATH # Default path is not setup

DATASET_ANNOTATION_PATH = os.path.join(SAVE_DATASET_PATH, "annotation")
IR_IMG_DIR_PATH = os.path.join(SAVE_DATASET_PATH, "ir_image")
DEPTH_IMG_DIR_PATH = os.path.join(SAVE_DATASET_PATH, "depth_image")

# Setup the checkpoint path
CHECKPOINT_PATH = ""
CHECKPOINT_PATH = os.path.join(PROJ_PATH, "checkpoint") if not CHECKPOINT_PATH else CHECKPOINT_PATH # Default path is not setup

MODEL_NAME = "ResnetCenterNet"

# Setup the data to be used for training (Depth images/ fusion of IR and Depth images)
DATA_LOADER_SWITCHER = {
    "depth": False,
    "fused": True,
}
DATA_LOADER = [[elem[0] for elem in DATA_LOADER_SWITCHER.items() if elem[1]][0]] [0]

# Setup the heatmap loss MSE/Logistic loss
LOSS_SWITHCER = {
    "MSE": False,
    "Logistic": True,
}
LOSS = [[elem[0] for elem in LOSS_SWITHCER.items() if elem[1]][0]] [0]


# Data Preprocessing
MAX_DEPTH_MAGNIFICATION = 6
MAX_IR_MAGNIFICATION = 3
MAX_ZOOM_OUT = 3
IMG_SHAPE = (320, 320)
BATCH_SIZE = 8
MAX_EPOCH = 3400
PRINT_FREQ = 200
SAVE_FREQ = 5

NUM_CLASSES = 1
STRIDE = 2

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 5e-4

ALPHA = 2
BETA = 4
THRESHOLD = 0.93

LAMBDA_HEATMAP = 10
LAMBDA_SIZE = 0.1
LAMBDA_OFFSET = 1

# INFRENCE
THRESHOLD_ACC = 0.3