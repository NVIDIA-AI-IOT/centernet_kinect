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
        
        print(f"Epoch: {epoch}\t"\
            f"Loss: {train_res['loss']:1.3f}\t"\
            f"Time: {train_res['time']:1.3f}\t")
        
        if ((epoch+1)%const.SAVE_FREQ == 0):
            valid_res = Run_Model(model_setup, False)

            print(f"Vaidation Epoch: {epoch}\t"\
                f"Vaidation Loss: {valid_res['loss']:1.3f}\t"\
                f"Vaidation Time: {valid_res['time']:1.3f}\t")
            model_setup.save()
    
    model_setup.save()
    print(f"Finished training at epoch {model_setup.epoch}")


if __name__ == '__main__':
    main()