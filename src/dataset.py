from pathlib import Path

# import torchvision.transforms as transforms
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

VALUE2CLASS = {
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


class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        key = lambda p: int(p.stem)
        self.img_paths = sorted(Path(img_dir).glob("*"), key=key)
        self.mask_paths = sorted(Path(mask_dir).glob("*"), key=key)

        self.gray_to_class = {
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

        self.lookup = np.zeros(256, dtype=np.int64)
        for gray, cls in self.gray_to_class.items():
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


if __name__ == "__main__":
    dataset = SegmentationDataset(img_dir="data/input", mask_dir="data/target")

# Использование:
# loader = DataLoader(SegDataset("data/input", "data/target"), batch_size=4, shuffle=True)
