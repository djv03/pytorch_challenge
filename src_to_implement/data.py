from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import pandas as pd
import torchvision as tv
import os

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


# class SolarDefectDataset(Dataset):
#     def __init__(self, csv_file, img_dir, transform=None):
#         self.data = pd.read_csv(csv_file, sep=";")
#         self.img_dir = img_dir
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         img_path = os.path.join(self.img_dir, os.path.basename(row['filename']))
#         image = Image.open(img_path).convert("RGB")
#         labels = torch.tensor([row['crack'], row['inactive']], dtype=torch.float32)
        
#         if self.transform:
#             image = self.transform(image)

#         return image, labels
    
    
class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    
    def __init__(self,data: pd.DataFrame, mode: str):
        self.data = data
        self.mode = mode.lower()
    
    # Common transforms
        if self.mode == 'train':
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomRotation(10),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        else:  # validation
            self.transform = tv.transforms.Compose([
                tv.transforms.ToPILImage(),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=train_mean, std=train_std)
            ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        row= self.data.iloc[index]
        img_path= row['filename']
        
        # print(f"[DEBUG] Current image path: {img_path}")
        
        # read image
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found:{img_path}")
        
        img= imread(img_path)
        
        # convert to the grayscale to RGB
        if len(img.shape)==2 or img.shape[2]==1:
            img= gray2rgb(img)
            
        # returning labels: two defect types
        label= torch.tensor([row['crack'],row['inactive']],dtype= torch.float32)
        
        # apply transformation
        img= self.transform(img)
        
        return img,label
