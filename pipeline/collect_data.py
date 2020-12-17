'''
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import cv2
import argparse
import numpy as np

from glob import glob
from pyk4a import PyK4A

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

# PROJ LIBRARY
import pipeline.constants as const

def parse_arguments():
    """
    Argument parser function for main.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number_of_frames',
                        type=int,
                        default=50,
                        help="Set the number of frames you wish to record")
    
    parser.add_argument('-s', '--save_path',
                        type=str,
                        default=const.SAVE_DATASET_PATH,
                        help="The path to save the dataset")

    args = parser.parse_args()
    return args

def setup_data_dir(save_path=const.SAVE_DATASET_PATH):
    """
    Set up the data directory to record the images

    :param save_path: full path to the data directory
    """
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        
    dirs = ["annotation", "casted_ir_image", "depth_image", "ir_image", "rgb_image"]
    for dir_name in dirs:
        tmp_path = os.path.join(save_path, dir_name)
        if not os.path.isdir(tmp_path):
            os.mkdir(tmp_path)

def capture_frames(iterations=50, save_path=const.SAVE_DATASET_PATH):
    """
    Capturn and store images from the camera

    :param iterations: the total number of frames to record
    :param save_path: full path to the data directory
    """
    ir_dir_path = os.path.join(save_path, "ir_image")
    depth_dir_path = os.path.join(save_path, "depth_image")
    rgb_dir_path = os.path.join(save_path, "rgb_image")
    casted_dir_path = os.path.join(save_path, "casted_ir_image")

    num_existing_frames = len(glob(os.path.join(ir_dir_path, "*.png")))
    num_existing_frames = 0 if num_existing_frames==0 else num_existing_frames+1

    # Load camera with default config
    k4a = PyK4A()
    k4a.start()
    
    print("Data Capture Starting!\n")
    for i in range(iterations):
        print(f"frame {i} outof {iterations}")

        capture = k4a.get_capture()
        ir_img = capture.ir
        depth_img = capture.depth
        rgb_img = capture.color

        frame_name = str(num_existing_frames+i).zfill(7) + ".png"

        # Store the images
        cv2.imwrite(os.path.join(ir_dir_path, frame_name), ir_img)
        cv2.imwrite(os.path.join(depth_dir_path, frame_name), depth_img)
        cv2.imwrite(os.path.join(rgb_dir_path, frame_name), rgb_img)
        cv2.imwrite(os.path.join(casted_dir_path, frame_name), ir_img.astype(np.uint8))
    
    print(f"IR images stored at: {ir_dir_path}")
    print(f"Depth images stored at: {depth_dir_path}")
    print(f"RGBA images stored at: {rgb_dir_path}")
    print(f"Casted IR images atored at: {casted_dir_path}")
    print("Data capture finished")

def main():
    args = parse_arguments()

    # Setup the savepath directories if they dont exist
    setup_data_dir(save_path=args.save_path)

    # Capture frames
    capture_frames(args.number_of_frames, save_path=args.save_path)

if __name__ == "__main__":
    main()