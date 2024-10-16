import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T

train_augs = T.Compose([T.RandomRotation(20), T.ToTensor()])

mnist_dataset = datasets.MNIST('MNIST/', download=True, train=True, transform=train_augs)


class CustomMnistDataset(Dataset):

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]

        if len(img.shape) == 3:
            return MixedTrainingMNISTDataset.process_single_pair(img)

        images_list = []
        for single_pair in img:
            processed = MixedTrainingMNISTDataset.process_single_pair(single_pair)
            images_list.append(processed)
        return torch.stack((images_list))

    @staticmethod
    def process_single_pair(pair):
        normal_img = Image.fromarray(pair[0].numpy(), mode="L")
        outlier_img = Image.fromarray(pair[1].numpy(), mode="L")
        normal_img = train_augs(normal_img)
        outlier_img = train_augs(outlier_img)
        return torch.stack((normal_img, outlier_img), dim=0).squeeze(1)


class MixedTrainingMNISTDataset(CustomMnistDataset):
    def __init__(self, normal_data, outliers_data):
        self.data = self.__shuffle(outliers_data, normal_data)

    def __shuffle(self, outliers_data, normal_data):

        outliers_shuffle_indices = torch.randint(0, len(outliers_data), (len(normal_data),))
        outliers_shuffled = outliers_data.dataset[outliers_data.indices][outliers_shuffle_indices]

        output = torch.stack((normal_data.dataset[normal_data.indices], outliers_shuffled), dim=1)
        return output


class MixedTestingMNISTDataset(CustomMnistDataset):
    def __init__(self, normal_data, outliers_data):
        self.data = torch.cat((normal_data, outliers_data), dim=0)
        normal_labels = torch.zeros((len(normal_data), ))
        outliers_labels = torch.ones((len(outliers_data), ))
        self.labels = torch.cat((normal_labels, outliers_labels), dim=0)

    def __getitem__(self, idx):
        features = self.data[idx]
        img = Image.fromarray(features.numpy(), mode="L")
        proc_img = train_augs(img)
        label = self.labels[idx]
        return proc_img, label

