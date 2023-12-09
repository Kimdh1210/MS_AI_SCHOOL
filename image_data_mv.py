import os
import shutil
import random

org_folder_path = "./Biscuit Wrappers Dataset"
train_folder_path = "./dataset/train/"
val_folder_path = "./dataset/val/"

# new create folder
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(val_folder_path, exist_ok=True)
sub_folder_list = [f for f in os.listdir(org_folder_path)
                  if os.path.isdir(os.path.join(org_folder_path, f))]
# print(sub_folder_list)
train_ratio = 0.9 # train 9 val 1
for folder in sub_folder_list :
    scr_folder = os.path.join(org_folder_path, folder)
    train_dst_folder = os.path.join(train_folder_path, folder)
    val_dst_folder = os.path.join(val_folder_path, folder)

    os.makedirs(train_dst_folder, exist_ok=True)
    os.makedirs(val_dst_folder, exist_ok=True)

    # subfolder image data get
    image_files_list = [f for f in os.listdir(scr_folder) if
                        f.lower().endswith(('.jpg', '.png', '.bmp', '.gif'))]

    random.shuffle(image_files_list)

    # train 90%
    num_train = int(len(image_files_list) * train_ratio)
    train_files = image_files_list[:num_train]

    for file in train_files :
        scr_path = os.path.join(scr_folder, file)
        dst_path = os.path.join(train_dst_folder, file)
        shutil.copy(scr_path, dst_path)

    val_files = image_files_list[num_train:]

    for file in val_files :
        scr_path = os.path.join(scr_folder, file)
        dst_path = os.path.join(val_dst_folder, file)
        shutil.copy(scr_path, dst_path)