from pathlib import Path

# import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

GRAY_TO_CLASS = {
    0: 0,
    25: 1,
    30: 2,
    35: 3,
    40: 4,
    45: 5,
    50: 6,
    55: 7,
    60: 8,
    65: 9,
    70: 10,
    75: 11,
    80: 12,
    85: 13,
    90: 14,
    95: 15,
    100: 16,
    105: 17,
    110: 18,
    115: 19,
    120: 20,
    125: 21,
    130: 22,
    135: 23,
    140: 24,
    145: 25,
    150: 26,
    155: 27,
    160: 28,
    165: 29,
    170: 30,
    175: 31,
    180: 32,
    185: 33,
    190: 34,
    195: 35,
    200: 36,
    205: 37,
    210: 38,
    215: 39,
}


CLASS_TO_GRAY = {
    0: 0,
    1: 25,
    2: 30,
    3: 35,
    4: 40,
    5: 45,
    6: 50,
    7: 55,
    8: 60,
    9: 65,
    10: 70,
    11: 75,
    12: 80,
    13: 85,
    14: 90,
    15: 95,
    16: 100,
    17: 105,
    18: 110,
    19: 115,
    20: 120,
    21: 125,
    22: 130,
    23: 135,
    24: 140,
    25: 145,
    26: 150,
    27: 155,
    28: 160,
    29: 165,
    30: 170,
    31: 175,
    32: 180,
    33: 185,
    34: 190,
    35: 195,
    36: 200,
    37: 205,
    38: 210,
    39: 215,
}


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        key = lambda p: int(p.stem)
        self.img_paths = sorted(Path(img_dir).glob("*"), key=key)
        self.mask_paths = sorted(Path(mask_dir).glob("*"), key=key)

        self.lookup = np.zeros(256, dtype=np.int64)
        for gray, cls in GRAY_TO_CLASS.items():
            if 0 <= gray < 256:
                self.lookup[gray] = cls

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, i):
        img = np.array(Image.open(self.img_paths[i]).convert("RGB"))
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img /= 255.0
        x = img

        mask_gray = np.array(Image.open(self.mask_paths[i]).convert("L"))
        mask_idx = self.lookup[mask_gray]
        mask = torch.from_numpy(mask_idx).long()
        y = mask

        return x, y
