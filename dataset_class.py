import os
import torch
from PIL import Image
from imutils import paths
from torch.utils.data import Dataset

class FaceMaskDataset(Dataset):
    """Face Mask Dataset - contains images of people with and without masks"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            transform: Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.paths = list(paths.list_images(root_dir))
        self.class_names = ['without_mask', 'with_mask']

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = Image.open(self.paths[idx]).convert("RGB")
        label_name = self.paths[idx].split(os.path.sep)[-2]
        
        label = 1 if label_name == self.class_names[1] else 0

        if self.transform:
            image = self.transform(image)

        return image, label