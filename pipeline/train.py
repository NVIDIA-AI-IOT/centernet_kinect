'''
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import argparse

from glob import glob

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

# PROJ LIBRARY
import pipeline.constants as const

from pipeline.utils import *
from pipeline.setup import ModelSetup
from model.run import Run_Model

def parse_arguments():
    """
    Argument parser function for main.py
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume',
                        type=bool,
                        default=False,
                        help="Set to True for resuming training")

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    if args.resume:
        model = get_model()
        model_setup = ModelSetup(load=model)
    else:
        model_setup = ModelSetup()

    print("Start Training...\n")
    for epoch in range(model_setup.epoch, model_setup.max_epoch):
        model_setup.epoch = epoch

        train_res = Run_Model(model_setup, True)
        
        print(f"Train Epoch: {epoch}\t"\
            f"Loss: {train_res['loss']:1.3f}\t"\
            f"Time: {train_res['time']:1.3f}\t")
        

        valid_res = Run_Model(model_setup, False)

        print(f"Vaidation Epoch: {epoch}\t"\
            f"Vaidation Loss: {valid_res['loss']:1.3f}\t"\
            f"Vaidation Time: {valid_res['time']:1.3f}\t")
        
        if ((epoch+1)%const.SAVE_FREQ == 0):
            model_setup.save()
    
    model_setup.save()
    print(f"Finished training at epoch {model_setup.epoch}")


if __name__ == '__main__':
    main()
