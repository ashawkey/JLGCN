import os
import sys
import glob
from torch.utils.data import Dataset, DataLoader, default_collate

import vision

def collate_fn_remove_None(batch):
    # filter out bad data in a batch
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

class imageFolder(Dataset):
    def __init__(self, root, logger, extension=["*.jpg", "*.png"], use_list=None):
        self.root = root
        self.log = logger
        self.extension = extension
        self.use_list = use_list
        
        self.log.info("Creating ImageFolder Dataset ...")

        if self.use_list is not None:
            with open(self.use_list, "r") as f:
                self.samples = f.readlines()
        self.samples = []
        for ext in self.extension:
            self.samples += glob.glob(os.path.join(self.root, ext))
        
        self.log.log1(f"root_path: {self.root}")
        self.log.log1(f"Length: {len(self.samples)}")

        self.log.info("Created ImageFolder.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name = self.samples[idx]
        return vision.imread(file_name)


