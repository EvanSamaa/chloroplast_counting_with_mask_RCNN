from skimage import data, feature, filters, segmentation
import helper
import cv2
from PIL import Image as image
import numpy as np
import os
import dataset as DS
# Import Mask RCNN

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


if __name__ == "__main__":
    file_dict, mask_dict = DS.get_mask_and_data_dicts()
    train_dict = {}
    val_dict = {}
    for i in range(0, 600):
        train_dict[i] = file_dict[i]
    for i in range(0, len(file_dict)-600):
        val_dict[i] = file_dict[i]

    config = Config()
    config.NAME = "chloroplast"
    training_dataset = DS.Image_Dataset(file_dict=train_dict, mask_dict=mask_dict)
    training_dataset.add_class("Julie_lab", 1, "chloroplast")
    training_dataset.prepare()
    valid_dataset = DS.Image_Dataset(file_dict=val_dict, mask_dict=mask_dict)
    valid_dataset.add_class("Julie_lab", 1, "chloroplast")
    valid_dataset.prepare()

    MODEL_DIR = "./trained_model/"
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    model.load_weights("../Julie_lab_data/mask_rcnn_coco.h5", by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
    model.train(training_dataset, valid_dataset,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='heads')
    print("head done")
    model.train(training_dataset, valid_dataset,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=10,
                layers="all")



