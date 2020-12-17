'''
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import time
import torch

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

# PROJ LIBRARY
import pipeline.constants as const

from pipeline.setup import ModelSetup
from model.utils import Summary

# Set the global device variable to cuda is GPU is avalible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Run_Model(model_setup: ModelSetup, train: bool):
    """
    Run either training or validation on the model

    :param model_setup: ModelSetup, model setup state
    :param train: bool, run training or validation
    """
    model_setup.model.to(DEVICE)
    if train:
        model_setup.model.train()
        dataset = model_setup.train_dataset
    else:
        model_setup.model.eval()
        dataset = model_setup.valid_dataset
    
    losses = Summary()
    total_time = Summary()
    start_time = time.time()

    for i, (images, ground_truths) in enumerate(dataset):
        images = images.to(DEVICE)
        ground_truths = ground_truths.to(DEVICE)

        preds = model_setup.model(images)
        loss = model_setup.criterion(preds, ground_truths)
        
        if torch.isnan(loss):
            print("NAN!")
            exit(-1)

        losses.update(loss.item())

        if train:
            model_setup.optim.zero_grad()
            loss.backward()
            model_setup.optim.step()
        
        if ((i+1)%const.PRINT_FREQ == 0):
            end_time = time.time()
            total_time.update(end_time - start_time)

            print(f"=== Epoch: {model_setup.epoch} [{i}/{len(dataset)}]\t"\
                    f"Loss: {losses.avg:1.3f}\t"\
                    f"Time: {end_time - start_time:1.3f} ===")
            start_time = time.time()
        
        del preds, images, ground_truths, loss
    
    end_time = time.time()
    total_time.update(end_time - start_time)

    return dict(
        loss=losses.avg,
        time=total_time.sum,
    )

def Run_Inference(model_setup: ModelSetup, image: torch.tensor):
    """
    Run either training or validation on the model

    :param model_setup: ModelSetup, model setup state
    :param train: bool, run training or validation
    """
    model_setup.model.to(DEVICE)
    model_setup.model.eval()
    
    image = image.to(DEVICE)
    with torch.no_grad():
        preds = model_setup.model(image)

    return preds