import cv2
import numpy as np
import os
from tqdm import tqdm
import logging


class TempleRingImageLoader(object):
    default_config = {
        "root_path": "../data/templeRing",
        "format": "png"
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Sequence image loader config: ")
        logging.info(self.config)

        self.dataset_basedir = self.config["root_path"]
        camera_intrinsic_file = os.path.join(self.dataset_basedir, "camera.txt")
        self.camera_intrinsic = self.load_camera_intrinsics(camera_intrinsic_file)

        self.img_id = 0
        self.img_lst = [f for f in os.listdir(self.config["root_path"]) if f.endswith(self.config["format"])]
        self.img_lst.sort(key=lambda fn: int(fn.split(".")[0]))

    def load_camera_intrinsics(self, camera_intrinsics_file):
        intrinsic = np.loadtxt(camera_intrinsics_file)
        return intrinsic

    def __getitem__(self, idx):
        file_path = os.path.join(self.config["root_path"], self.img_lst[idx])
        img = cv2.imread(file_path)
        return img

    def __iter__(self):
        return self

    def __next__(self):
        if self.img_id < len(self):
            img = self.__getitem__(self.img_id)

            self.img_id += 1

            return img
        raise StopIteration()

    def __len__(self):
        return len(self.img_lst)
