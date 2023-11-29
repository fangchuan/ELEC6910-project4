import os
import sys
sys.path.append('.')
sys.path.append('..')

import cv2
import numpy as np
import os
from tqdm import tqdm
import logging
from utils.PinholeCamera import PinholeCamera

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

        # read images
        self.img_id = 0
        self.img_path_lst = [f for f in os.listdir(self.config["root_path"]) if f.endswith(self.config["format"])]
        self.img_path_lst.sort(key=lambda fn: int(fn.split(".")[0]))
        # print(self.img_path_lst)
        self.img_lst = []
        for idx in range(len(self.img_path_lst)):
            file_path = os.path.join(self.config["root_path"], self.img_path_lst[idx])
            img = cv2.imread(file_path)
            self.img_lst.append(img)

        camera_intrinsic_file = os.path.join(self.dataset_basedir, "camera.txt")
        self.camera_intrinsic = self.load_camera_intrinsics(camera_intrinsic_file)
        img_w, img_h = self.img_lst[0].shape[1], self.img_lst[0].shape[0]
        self.cam = PinholeCamera(width=img_w, height=img_h, fx=self.camera_intrinsic[0, 0], fy=self.camera_intrinsic[1, 1],
                                cx=self.camera_intrinsic[0, 2], cy=self.camera_intrinsic[1, 2])
        print(self.cam)

    def load_camera_intrinsics(self, camera_intrinsics_file):
        intrinsic = np.loadtxt(camera_intrinsics_file)
        return intrinsic

    def __getitem__(self, idx):
        return {'image':self.img_lst[idx], 'image_name':self.img_path_lst[idx]}

    def __iter__(self):
        return self

    def __next__(self):
        if self.img_id < len(self):
            img = self.img_lst[self.img_id]
            image_name = self.img_path_lst[self.img_id]
            self.img_id += 1

            return {'image':img, 'image_name':image_name}
        raise StopIteration()

    def __len__(self):
        return len(self.img_path_lst)
