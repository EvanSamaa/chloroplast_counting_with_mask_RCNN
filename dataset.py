from mrcnn import utils
import numpy as np
from PIL import Image
from os import listdir
from mrcnn.config import Config
class Image_Dataset(utils.Dataset):
    def __init__(self, class_map=None, file_dict=None, mask_dict=None):
        super().__init__(class_map)
        self.file_dict = file_dict  # stores a dictionary of form [number:file_name with relative path]
        self.mask_dict = mask_dict
        for i in file_dict.keys():
            self.add_image("Julie_lab", i, file_dict[i])

    def load_image(self, image_id):
        file_name = self.file_dict[image_id]
        image = np.asarray(Image.open(file_name))
        rtv = np.zeros((image.shape[0], image.shape[1], 3))
        rtv[:, :, 0] = image
        rtv[:, :, 1] = image
        rtv[:, :, 2] = image
        return rtv
    def load_mask(self, image_id):
        rtv = np.zeros((1023, 1024, 1))
        classes = [0]
        masks_names = self.mask_dict[image_id]
        for mask_name in masks_names:
            mask = np.asarray(Image.open(mask_name))
            rtv = np.concatenate((rtv,mask.reshape(1023, 1024, 1)), axis=2)
            classes.append(1)
        classes = np.array(classes)
        return rtv.astype(np.bool), classes.astype(np.int32)

def get_mask_and_data_dicts(path_to_data="../Julie_lab_data/Full Dataset Processed/"):
    file_dict = {}
    mask_dict = {}
    data_dir = path_to_data + "Data/"
    mask_dir = path_to_data + "Masks/"
    file_list = sorted(listdir(data_dir))[1:]
    mask_list = sorted(listdir(mask_dir))[1:]

    # generate the dictionary that relate an image to its masks
    for i in range(0, len(file_list)):
        file_dict[i] = data_dir + file_list[i]
        file_name = file_list[i].split(".")[0]
        masks = []
        for mask in mask_list:
            mask_name = mask[:-4].split("_mask_")[0]
            if mask_name == file_name:
                masks.append(mask_dir + mask)
        mask_dict[i] = masks
    return file_dict, mask_dict


