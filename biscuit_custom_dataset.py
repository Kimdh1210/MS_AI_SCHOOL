import os
import glob
from torch.utils.data import Dataset
from PIL import Image
class BiscuitDataset(Dataset) :

    def __init__(self, path, transform=None):
        folders = os.listdir(path)
        self.labels_dict = {}
        for i , folder in enumerate(folders) :
            self.labels_dict[folder] = i
        # path -> ./dataset/train/
        self.path_list = glob.glob(os.path.join(path,"*","*.jpg" ))
        self.transform = transform

    def __getitem__(self, item):
        image_path = self.path_list[item]
        image = Image.open(image_path).convert("RGB")

        folder_name = image_path.split("\\")[1]
        label = self.labels_dict[folder_name]

        # aug
        if self.transform is not None :
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.path_list)

# test = BiscuitDataset("./dataset/val/", transform=None)
# for i in test :
#     print(i)