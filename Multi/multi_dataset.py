import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class SievesDataset(Dataset):
    def __init__(self, image_list, image_dir, mask_dir, transform=None):
        self.images = image_list
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        #Load image and corresponding mask
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.png"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # Preprocess mask (for multiclass)
        mask[mask == 127.0] = 1
        mask[mask == 255.0] = 2

        #Data augmentation
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

