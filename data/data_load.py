# -*- Coding: utf-8 -*-
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np

from data.autoaugment import ImageNetPolicy

Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class supImgDataset(Dataset):
    """
    Input: [{"img":, "label": }, ..]
    """

    def __init__(self, data_path, img_size, crop_size):
        super(supImgDataset, self).__init__()
        self._data_path = data_path
        self._img_size = img_size
        self._crop_size = crop_size
        self._trans_train = transforms.Compose([
            transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
            transforms.CenterCrop(self._crop_size),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor()
        ])

        self._data_list = np.load(self._data_path, allow_pickle=True)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, item):
        img = Image.open(self._data_list[item]["img"])
        img = self._trans_train(img)
        target = self._data_list[item]["label"]

        return img, target
        # return target


class unsupImgDataset(Dataset):
    """
    Input: [{"ori_img":, "linked_img":, "aug_img":, }]
    """

    def __init__(self, data_path, img_size, crop_size):
        super(unsupImgDataset, self).__init__()
        self._data_path = data_path
        self._img_size = img_size
        self._crop_size = crop_size
        self.trans = transforms.Compose([
            transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
            transforms.CenterCrop(self._crop_size),
            transforms.ToTensor()
        ])
        self._data_list = np.load(self._data_path, allow_pickle=True)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, item):
        ori_img = Image.open(self._data_list[item]["ori_img"])
        linked_img = Image.open(self._data_list[item]["linked_img"])
        aug_img = Image.open(self._data_list[item]["aug_img"])

        return self.trans(ori_img), self.trans(linked_img), self.trans(aug_img)

        # return self._data_list[item]["ori_img"]


class testDataset(Dataset):
    def __init__(self, data_path, img_size, crop_size):
        super(testDataset, self).__init__()
        self._data_path = data_path
        self._img_size = img_size
        self._crop_size = crop_size
        self._trans_test = transforms.Compose([
            transforms.ColorJitter(64.0 / 255, 0.75, 0.25, 0.04),
            transforms.CenterCrop(self._crop_size),
            transforms.ToTensor()
        ])
        self._data_list = np.load(self._data_path, allow_pickle=True)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, item):
        img = Image.open(self._data_list[item]["img"])
        img = self._trans_train(img)
        target = self._data_list[item]["label"]

        return img, target


class wsiDataset(Dataset):
    def __init__(self):
        super(wsiDataset, self).__init__()
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


if __name__ == "__main__":
    supdataset = supImgDataset(
        "/home/qianslab/yangwenhui/TUMOR/TUMOR_data_processed/bag_files/data_enrichment/Labeled_Trainset_path_with_9750_samples.npy",
        256, 224)
    unsupdataset = unsupImgDataset(
        "/home/qianslab/yangwenhui/TUMOR/TUMOR_data_processed/bag_files/data_enrichment/Unlabeled_Trainset_path_with_780000_.npy",
        256, 224)

    supdataloader = DataLoader(supdataset, batch_size=100, shuffle=True)
    unsupdataloader = DataLoader(unsupdataset, batch_size=1, shuffle=True)

    iterl = iter(supdataloader)
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    c = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    b = iter(a)
    d = iter(c)
    print(len(c) // len(a))
    for j in range(2):
        for i in range(len(a)):
            # print(i)
            print(next(b))
            print(next(d))
        b = iter(a)
        # print(next(iter(supdataloader)))
        # print(next(iterl))
        # print(next(iter(unsupdataloader)))

    # img = next(iter(unsupdataloader))[0][0]
    # print(len(img))
    # print(img.min())
    # print(img)
    # plt.imshow(img.permute(1,2,0))
    # plt.show()
