from torch.utils.data import Dataset
import cv2
import os
import sys
import random
import numpy as np
from PIL import Image

class ImageRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class DFDCDatatset(Dataset):
    def __init__(self, root_path, list_file, transform=None):
        self.root_path = root_path
        self.list_file = list_file
        self.transform = transform

        self._parse_list()

    def _load_image(self, image_path):
        return Image.open(image_path).convert('RGB')

    def _parse_list(self):
        self.image_list = [ImageRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.image_list[index]

        image = self._load_image(os.path.join(self.root_path, record.path))

        if self.transform is not None:
            image = self.transform(image)

        return image, record.label

    def __len__(self):
        return len(self.image_list)