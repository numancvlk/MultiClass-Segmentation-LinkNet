# LIBRARIES
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, imagesDir, masksDir, classDictPath, transform=None):
        self.imagesDir = imagesDir                # Görüntülerin bulunduğu klasör
        self.masksDir = masksDir                  # Maskelerin bulunduğu klasör
        self.transform = transform
        self.images = os.listdir(self.imagesDir)  # Görsel dosya isimlerini listele

        class_dict = pd.read_csv(classDictPath)

        self.colorToLabel = {
            (row["r"], row["g"], row["b"]): idx for idx, row in class_dict.iterrows()
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        imgPath = os.path.join(self.imagesDir, self.images[index])
        maskPath = os.path.join(self.masksDir, self.images[index].replace(".png", "_L.png"))

        image = np.array(Image.open(imgPath).convert("RGB"))
        maskRGB = np.array(Image.open(maskPath).convert("RGB"))

        mask = np.zeros((maskRGB.shape[0], maskRGB.shape[1]), dtype=np.uint8)

        for color, label in self.colorToLabel.items():
            mask[(maskRGB == color).all(axis=2)] = label

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
