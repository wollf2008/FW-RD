from torchvision import transforms
from PIL import Image
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import numpy as np

def get_data_transforms(size, isize): #？？？
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms



class CamelyonDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.img_path = root
        self.transform = transform
        # load dataset
        self.img_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path) 
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_paths.sort()
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        return img_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path,  label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label, img_type


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train') #指定train图片集的路径，root是：/mvtec/物体名称
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
            self.background_path = os.path.join(root, 'background_mask')
        self.transform = transform
        self.gt_transform = gt_transform
        self.background_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.background_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        background_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                background_paths = glob.glob(os.path.join(self.background_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                background_tot_paths.extend(background_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                background_paths = glob.glob(os.path.join(self.background_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                background_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                background_tot_paths.extend(background_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths) == len(background_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, background_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, background, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.background_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        background = Image.open(background).convert('L')
        background = self.background_transform(background)

        assert img.size()[1:] == gt.size()[1:] == background.size()[1:], "image.size != gt.size !!!"

        return img, gt, background, label, img_type