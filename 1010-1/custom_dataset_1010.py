import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class MyCustomDataset(Dataset) :
    def __init__(self, path, transform=None):
        # path = "./dataset/train/"
        self.path = glob.glob(os.path.join(path, "*", "*.png"))
        # path -> [./dataset/train/STFT/xxxx.png, ...]
        self.transform = transform
        self.label_dict = {"MelSepctrogram" : 0,
                           "STFT" : 1,
                           "waveshow" : 2,
                           }

    def __getitem__(self, item):
        image_path = self.path[item]
        # image read
        image = Image.open(image_path).convert("RGB")

        # label
        folder_name = image_path.split("\\")[1]
        label = self.label_dict[folder_name]
        print(folder_name, label)
        #['./dataset/train', 'MelSepctrogram',
        # 'pop.00095_augmented_stretch.png']

        # ./dataset/train\MelSepctrogram\country.00066_mel_spec.png
        pass

    def __len__(self):
        pass

test = MyCustomDataset("./dataset/train/", transform=None)
for i in test :
    pass