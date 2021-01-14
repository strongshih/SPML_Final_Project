from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import glob
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import numpy as np
import os
from PIL import Image

name2label = {"airplane":0, "automobile":1, "bird":2, "cat":3, "deer":4, "dog":5, "frog":6, "horse":7, "ship":8, "truck":9}
label2name = {v: k for k, v in name2label.items()}

class myCIFAR10(Dataset):
    def __init__(self, root_name: str = "src", normalize = False, image_type="png"):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        # if root_name != "src" and root_name != "adv_imgs":
        #     raise NotImplementError

        self.img_paths = []
        for path in glob.glob("./"+root_name+"/*/*."+image_type):
            self.img_paths.append(path)

        #cifar 10
        self.mean_rgb = (0.4914, 0.4822, 0.4465)
        self.std_rgb = (0.2023, 0.1994, 0.2010)
        
        
        if normalize:
            self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.mean_rgb,std=self.std_rgb)])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])
        
    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        img_path = self.img_paths[index]
        img_PIL = Image.open(img_path)
        img_PIL = img_PIL.resize((32, 32), Image.ANTIALIAS)
        img_tensor = self.transforms(img_PIL)
        return img_tensor, name2label[img_path.split("/")[2]]


    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.img_paths)
