from skimage import data, feature, filters, segmentation
import helper
import cv2
from PIL import Image as image
import numpy as np
import os
import dataset as DS
# Import Mask RCNN
import random
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

class InferenceConfig(Config ):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = "julie lab"

class Inference():
    def __init__(self, dataset, MODEL_DIR="./trained_model/chloroplast20200523T2344_mask_rcnn_chloroplast_0010.h5"):
        self.inference_config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference",
                          config=self.inference_config,
                          model_dir=MODEL_DIR)
        self.model.load_weights(MODEL_DIR, by_name=True)
        self.dataset_val = dataset
    def eval(self, img=None):
        image_id = random.choice(self.dataset_val.image_ids)
        img = self.dataset_val.load_image(image_id)
        results = self.model.detect([img], verbose=1)

        r = results[0]
        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                    ['BG', 'chloroplast'], r['scores'], figsize=(8, 8))

if __name__ == "__main__":
    mode = "valid"
    file_dict, mask_dict = DS.get_mask_and_data_dicts()
    train_dict = {}
    val_dict = {}
    for i in range(0, 600):
        train_dict[i] = file_dict[i]
    for i in range(0, len(file_dict)-600):
        val_dict[i] = file_dict[i]

    if mode == "train":
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
        model.load_weights("./julie_lab_images_0/mask_rcnn_coco.h5", by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
        # model.train(training_dataset, valid_dataset,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=2,
        #             layers='heads')
        # print("head done")
        model.train(training_dataset, valid_dataset,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=10,
                    layers="all")
    else:
        valid_dataset = DS.Image_Dataset(file_dict=val_dict, mask_dict=mask_dict)
        valid_dataset.add_class("Julie_lab", 1, "chloroplast")
        valid_dataset.prepare()
        inference = Inference(valid_dataset)
        for i in range(0, 10):
            inference.eval()
