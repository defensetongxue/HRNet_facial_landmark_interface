import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np

class CUSTOM(data.Dataset):
    def __init__(self, data_root,transform=None):
        # specify annotation file for dataset
        

        self.transform = transform
        self.data_root = data_root
        self.input_size = (224,224)
        self.output_size = (56,56) # heatmap size,should be 1/4 of image_size
        self.scale_factor = 0.25

        self.index_list=[f for f in sorted(os.listdir(data_root))]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  f"{self.index_list[idx]}")

        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        
        return img, torch.zeros((1,1)), self.index_list[idx]




if __name__ == '__main__':
    pass