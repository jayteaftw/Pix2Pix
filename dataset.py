from PIL import Image
import numpy as np
import os 
from torch.utils.data import Dataset
import config

class MapDataset(Dataset):
    def __init__(self, root_directory) -> None:
        self.root_directory = root_directory
        self.list_files = os.listdir(self.root_directory)
        print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_directory, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :600, :]
        target_image = image[:, 600:, :]

        augmentations = config.both_transform(image=input_image, image0=target_image )
        input_image, target_image = augmentations["image"], augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_target(image=target_image)["image"]

        return input_image, target_image





        