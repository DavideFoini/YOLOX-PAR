import os
import scipy
import torch
import torchvision
from tabulate import tabulate
from torch.utils.data import Dataset
from torchvision.io import read_image


class Train(Dataset):
    def __init__(self, dataset_path, transform=torchvision.transforms.Resize((250, 130), antialias=True), device='cpu', multitask = False, only_gender=False):
        self.device = device
        mat = scipy.io.loadmat(dataset_path + '/annotation.mat', struct_as_record=True, simplify_cells=True)
        self.imgs_path = dataset_path + '/images'
        self.labels_list = mat['attributes'].tolist()
        self.imgs_names = mat['train_images_name'].tolist()
        self.labels = torch.from_numpy(mat['train_label'])
        self.transform = transform
        self.multitask = multitask
        self.only_gender = only_gender

    def __len__(self):
        return len(self.imgs_names)

    def __getitem__(self, idx):
        img_name = self.imgs_names[idx]
        img_path = os.path.join(self.imgs_path, img_name)
        image = read_image(img_path)
        # preprocess
        image = self.transform(image)
        # image = even_img(image)
        image = torch.div(image, 255).to(self.device)

        if not self.multitask:
            label = self.labels[idx]
            if self.only_gender:
                return image, label[0].float().to(self.device)
            else:
                return image, label.float().to(self.device)
        else:
            gender = self.labels[idx][0]
            label = torch.cat((self.labels[idx][1:4], self.labels[idx][7:13], self.labels[idx][15:]))
            pov = self.labels[idx][4:7]
            sleeve = self.labels[idx][13:15]
            return image, gender.float().to(self.device), pov.float().to(self.device), sleeve.float().to(self.device), label.float().to(self.device)

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
class Test(Dataset):
    def __init__(self, dataset_path, transform=torchvision.transforms.Resize((250, 130), antialias=True), device='cpu', multitask=False, only_gender=False):
        self.device = device
        mat = scipy.io.loadmat(dataset_path + '/annotation.mat', struct_as_record=True, simplify_cells=True)
        self.imgs_path = dataset_path + '/images'
        self.labels_list = mat['attributes'].tolist()
        self.imgs_names = mat['test_images_name'].tolist()
        self.labels = torch.from_numpy(mat['test_label'])
        self.transform = transform
        self.multitask = multitask
        self.only_gender = only_gender

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
            if self.only_gender:
                return image, label[0].float().to(self.device)
            else:
                return image, label.float().to(self.device)
        else:
            gender = self.labels[idx][0]
            label = torch.cat((self.labels[idx][1:4], self.labels[idx][7:13], self.labels[idx][15:]))
            pov = self.labels[idx][4:7]
            sleeve = self.labels[idx][13:15]
            return image, gender.float().to(self.device), pov.float().to(self.device), sleeve.float().to(self.device), label.float().to(self.device)


def even_img(img):
    odd = False
    h, w = img.size(dim=1), img.size(dim=2)
    if h%2 != 0:
        h -= 1
        odd = True
    if w%2 != 0:
        w -= 1
        odd = True
    if odd:
        transform = torchvision.transforms.Resize((h, w), antialias=True)
        img = transform(img)
    return img