import os
import scipy
import torch
import torchvision
from tabulate import tabulate
from torch.utils.data import Dataset
from torchvision.io import read_image


# TODO implement
class PETA(Dataset):
    def __init__(self, dataset_path, transform=torchvision.transforms.Resize((250, 130), antialias=True), device='cpu', multitask = False):
        self.device = device
        self.imgs_path = dataset_path + '/files'
        #self.labels_list =
        #self.imgs_names =
        #self.labels = torch.from_numpy(mat['train_label'])
        self.transform = transform
        self.multitask = multitask

    def __len__(self):
        return len(self.imgs_names)

    def __getitem__(self, idx):
        img_name = self.imgs_names[idx]
        img_path = os.path.join(self.imgs_path, img_name)
        image = read_image(img_path)
        # preprocess
        image = self.transform(image)
        image = torch.div(image, 255).to(self.device)

        if not self.multitask:
            label = self.labels[idx]
            return image, label.float().to(self.device)
        else:
            gender = self.labels[idx][0]
            label = self.labels[idx][1:]
            return image, gender.float().to(self.device), label.float().to(self.device)

    def check_balance(self):
        labels_num = [0] * len(self.labels_list)
        for label in self.labels:
            for i in range(0, len(label)):
                if label[i] == 1: labels_num[i] += 1

        for i in range(0, len(labels_num)):
            labels_num[i] = "{} - {}%".format(str(labels_num[i]), str(labels_num[i] / len(self.labels) * 100))

        table = zip(self.labels_list, labels_num)
        table = tabulate(table, headers=("Label", "Rateo"), tablefmt='fancy_grid')
        print(table)