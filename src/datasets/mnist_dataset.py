import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms as T

train_augs = T.Compose([T.RandomRotation(20), T.ToTensor()])

mnist_dataset = datasets.MNIST('MNIST/', download=True, train=True, transform=train_augs)


class CustomMNISTDataset(Dataset):
    def __init__(self, sample_size, label, exclude=False):
        if exclude:
            mask = mnist_dataset.targets != label
        else:
            mask = mnist_dataset.targets == label
        self.sample_size = min(sample_size, len(mask))
        self.data = mnist_dataset.data[mask][:self.sample_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        if len(img.shape) == 2:
            img = Image.fromarray(img.numpy(), mode="L")
            img = train_augs(img)
            return img

        images_list = []
        for single_img in img:
            img_img = Image.fromarray(single_img.numpy(), mode="L")
            img_img = train_augs(img_img)
            images_list.append(img_img)
        return torch.stack((images_list))

