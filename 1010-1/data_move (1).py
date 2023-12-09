"""
class number 3 (MelSepctrogram, STFT, waveshow)
dataset 9:1
  train
    - MelSepctrogram
       - image.png ....
    - STFT
       - image.png ...
    - waveshow
       - image.png ...
  val
    - MelSepctrogram
       - image.png ....
    - STFT
       - image.png ...
    - waveshow
       - image.png ...

dataset
 class number 30
 train
  - MelSepctrogram_bluse
  - MelSepctrogram_classical
  - MelSepctrogram_country
   .....
  - STFT_bluse
    ....
  - waveshow_bluse
    ....

"""
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

data_path = "./final_data"
Mel_data_list = glob.glob(os.path.join(data_path, "MelSepctrogram", "*", "*.png"))
STFT_data_list = glob.glob(os.path.join(data_path, "STFT", "*", "*.png"))
waveshow_data_list = glob.glob(os.path.join(data_path, "waveshow", "*", "*.png"))

"""
# train_test_split - 3 (MelSpctrogram, STFT, waveshow)
"""
mel_train_list, mel_val_list = train_test_split(Mel_data_list,
                                                test_size=0.1, random_state=777)
stft_train_list, stft_val_list = train_test_split(STFT_data_list,
                                                  test_size=0.1, random_state=777)
waveshow_train_list, waveshow_val_list = train_test_split(waveshow_data_list,
                                                test_size=0.1, random_state=7777)

for mel_train_lists in mel_train_list :
    os.makedirs("./dataset/train/MelSepctrogram/", exist_ok=True)
    # ./final_data\MelSepctrogram\pop\pop.00018_augmented_noise.png
    image_name = os.path.basename(mel_train_lists)
    shutil.copy(mel_train_lists, f"./dataset/train/MelSepctrogram/{image_name}")
for mel_val_lists in mel_val_list :
    os.makedirs("./dataset/val/MelSepctrogram/", exist_ok=True)
    # ./final_data\MelSepctrogram\pop\pop.00018_augmented_noise.png
    image_name = os.path.basename(mel_val_lists)
    shutil.copy(mel_val_lists, f"./dataset/val/MelSepctrogram/{image_name}")

for stft_train_lists in stft_train_list :
    os.makedirs("./dataset/train/STFT/", exist_ok=True)
    # ./final_data\MelSepctrogram\pop\pop.00018_augmented_noise.png
    image_name = os.path.basename(stft_train_lists)
    shutil.copy(stft_train_lists, f"./dataset/train/STFT/{image_name}")
for stft_val_lists in stft_val_list :
    os.makedirs("./dataset/val/STFT/", exist_ok=True)
    # ./final_data\MelSepctrogram\pop\pop.00018_augmented_noise.png
    image_name = os.path.basename(stft_val_lists)
    shutil.copy(stft_val_lists, f"./dataset/val/STFT/{image_name}")

for waveshow_train_lists in waveshow_train_list :
    os.makedirs("./dataset/train/waveshow/", exist_ok=True)
    # ./final_data\MelSepctrogram\pop\pop.00018_augmented_noise.png
    image_name = os.path.basename(waveshow_train_lists)
    shutil.copy(waveshow_train_lists, f"./dataset/train/waveshow/{image_name}")
for waveshow_val_lists in waveshow_val_list :
    os.makedirs("./dataset/val/waveshow/", exist_ok=True)
    # ./final_data\MelSepctrogram\pop\pop.00018_augmented_noise.png
    image_name = os.path.basename(waveshow_val_lists)
    shutil.copy(waveshow_val_lists, f"./dataset/val/waveshow/{image_name}")

print("ok ~~~")