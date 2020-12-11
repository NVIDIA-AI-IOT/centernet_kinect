import os
import sys
import torch
from torch.utils.data import DataLoader

# PROJ ROOT DIR
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = os.path.join(DIR_PATH, os.path.pardir)
sys.path.append(ROOT_PATH)

# PROJ LIBRARY
import pipeline.constants as const

from model.centerNet import Resnet18FeatureExtractor
from model.utils import Criterion
from dataLoader.dataLoader import ProjDataLoader
from dataLoader.fusedDataLoader import FusedProjDataLoader

class ModelSetup(object):
    """
    Model setup class to configure model and dataloader
    """

    model_data_loader_switcher = {
        "depth": ProjDataLoader,
        "fused": FusedProjDataLoader,
    }

    def __init__(self, dataset_path=const.DATASET_PATH, json_annotation_path=const.JSON_ANNOTATION_PATH,\
                    ir_img_dir_path=const.IR_IMG_DIR_PATH, depth_img_dir_path=const.DEPTH_IMG_DIR_PATH,\
                    num_classes=const.NUM_CLASSES, load=None, pretrained=True, batch_size=const.BATCH_SIZE,\
                    checkpoint_dir=const.CHECKPOINT_PATH, infer=False):
        if infer and not os.path.isfile(load):
            print(f"no such file exists: {load}")
            exit(-1)
            
        print("Setting up model...")
        self.dataset_path = dataset_path
        self.json_annotation_path = json_annotation_path
        self.ir_img_dir_path = ir_img_dir_path
        self.depth_img_dir_path = depth_img_dir_path
        self.num_classes = num_classes
        self.checkpoint_dir = checkpoint_dir
        self.loss = const.LOSS

        self.input_format = const.DATA_LOADER

        if load:
            print(f"Loading Model from: {load}")
            checkpoint = torch.load(load, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.epoch = checkpoint["epoch"]+1
            self.input_format = checkpoint["input_format"]
            self.model_name = checkpoint["model_name"]
            self.num_classes = checkpoint["num_classes"]
            self.model = Resnet18FeatureExtractor(num_classes=self.num_classes)
            self.model.load_state_dict(checkpoint["model"])
            self.optim = torch.optim.Adam(self.model.parameters(), lr=const.LEARNING_RATE, weight_decay=const.WEIGHT_DECAY)
            # self.optim = torch.optim.Adam(self.model.parameters(), lr=const.LEARNING_RATE)
            self.optim.load_state_dict(checkpoint["optim"])

            if torch.cuda.is_available():
                for state in self.optim.state.values():
                    for k,v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            print("Loading model finished!")
        
        else:
            print(f"Creating model: {const.MODEL_NAME}")
            self.epoch = 0
            self.model_name = const.MODEL_NAME
            self.num_classes = num_classes
            self.model = Resnet18FeatureExtractor(num_classes=self.num_classes, pretrained=pretrained)
            self.optim = torch.optim.Adam(self.model.parameters(), lr=const.LEARNING_RATE, weight_decay=const.WEIGHT_DECAY)
            # self.optim = torch.optim.Adam(self.model.parameters(), lr=const.LEARNING_RATE)
            print("Finished creaing model!")
        
        self.criterion = Criterion

        model_data_loader = self.model_data_loader_switcher[const.DATA_LOADER]

        load_train = model_data_loader(dataset_path=self.dataset_path,
                            json_annotation_path=self.json_annotation_path,
                            train=True,
                            ir_img_dir_path=self.ir_img_dir_path,
                            depth_img_dir_path=self.depth_img_dir_path)
        self.train_dataset = DataLoader(
            load_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=batch_size,
            pin_memory=True,   
        )

        load_valid = model_data_loader(dataset_path=self.dataset_path,
                            json_annotation_path=self.json_annotation_path,
                            train=False,
                            ir_img_dir_path=self.ir_img_dir_path,
                            depth_img_dir_path=self.depth_img_dir_path)
        self.valid_dataset = DataLoader(
            load_valid,
            batch_size=batch_size,
            shuffle=True,
            num_workers=batch_size,
            pin_memory=True,   
        )
        self.max_epoch = const.MAX_EPOCH
        print("Model setup finished!\n\n")

    def save(self):
        save_path = self.checkpoint_dir
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        
        # save_path = os.path.join(save_path, f"{self.model_name}_{self.input_format}.pth")
        save_path = os.path.join(save_path, f"{self.loss}_{self.model_name}_{self.input_format}.pth")

        torch.save(dict(
            model_name=self.model_name,
            model=self.model.state_dict(),
            num_classes=self.num_classes,
            optim=self.optim.state_dict(),
            input_format=self.input_format,
            epoch=self.epoch,
        ), save_path)
        print(f"Model Saved at: {save_path}")