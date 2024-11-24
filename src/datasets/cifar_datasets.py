import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomCIFARDataset(Dataset):
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]

        if len(img.shape) == 4:
            return self.process_single_pair(img)

        images_list = []
        for single_pair in img:
            processed = self.process_single_pair(single_pair)
            images_list.append(processed)
        return torch.stack((images_list))

    def process_single_pair(self, pair):
        normal_img = pair[0]
        outlier_img = pair[1]

        normal_img = Image.fromarray(normal_img.numpy().astype("uint8"))
        outlier_img = Image.fromarray(outlier_img.numpy().astype("uint8"))

        normal_img = self.transform(normal_img)
        outlier_img = self.transform(outlier_img)

        output = torch.stack((normal_img, outlier_img), dim=0).squeeze(1)
        return output


class MixedTestingCIFARDataset(CustomCIFARDataset):
    def __init__(self, normal_data, outliers_data, transform=None):

        self.data = np.concatenate((normal_data, outliers_data), axis=0)
        if transform:
            self.transform = transform

        normal_labels = torch.zeros((len(normal_data), ))
        outliers_labels = torch.ones((len(outliers_data), ))
        self.labels = torch.cat((normal_labels, outliers_labels), dim=0)

    def __getitem__(self, idx):
        def process_img(item):
            return self.transform(item).reshape(3, 32, 32)

        img = self.data[idx]
        if len(img.shape) == 3:
            result = process_img(img)
        else:
            result = torch.empty(size=(len(img), 3, 32, 32))
            for i, item in enumerate(img):
                result[i] = process_img(item)
        label = self.labels[idx]
        return result, label

