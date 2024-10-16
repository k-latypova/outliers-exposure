import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T

train_augs = T.Compose([T.RandomRotation(20), T.ToTensor()])

mnist_dataset = datasets.MNIST(
    "MNIST/", download=True, train=True, transform=train_augs
)


class CustomMixedMNISTDataset(Dataset):
    def __init__(
        self,
        normal_sample_size,
        outlier_sample_size,
        normal_label,
        outlier_label,
        exclude=False,
    ):
        if exclude:
            mask = mnist_dataset.targets != normal_label
        else:
            mask = mnist_dataset.targets == normal_label
        self.__normal_sample_size = min(normal_sample_size, len(mask))
        normal_data = mnist_dataset.data[mask]
        # self.__normal_data = mnist_dataset.data[mask][:self.__normal_sample_size]
        # self.__normal_data = normal_data[normal_data_indices]
        mask = mnist_dataset.targets == outlier_label
        self.__outliers_sample_size = min(outlier_sample_size, len(mask))
        outliers_data = mnist_dataset.data[mask]
        # self.__outliers_data = mnist_dataset.data[mask][:self.__outliers_sample_size]
        self.data = self.__shuffle(outliers_data, normal_data)
        # self.data = self.__mix()

    def __mix(self):
        tensor1_repeated = self.__normal_data.unsqueeze(1).repeat(
            1, self.__outliers_sample_size, 1, 1
        )

        # Reshape and repeat tensor2 to shape (n, m, 28, 28)
        tensor2_repeated = self.__outliers_data.unsqueeze(0).repeat(
            self.__normal_sample_size, 1, 1, 1
        )

        # Stack the repeated tensors along a new dimension
        output = torch.stack((tensor1_repeated, tensor2_repeated), dim=2)

        # Reshape the output tensor to shape (m*n, 2, 28, 28)
        output = output.view(
            self.__normal_sample_size * self.__outliers_sample_size, 2, 28, 28
        )
        return output

    def __shuffle(self, outliers_data, normal_data):
        normal_digits = normal_data[: self.__normal_sample_size]

        outliers_shuffle_indices = torch.randint(
            0, self.__outliers_sample_size, (self.__normal_sample_size,)
        )
        outliers_shuffled = outliers_data[: self.__outliers_sample_size][
            outliers_shuffle_indices
        ]

        output = torch.stack((normal_digits, outliers_shuffled), dim=1)
        return output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]

        if len(img.shape) == 3:
            return CustomMixedMNISTDataset.process_single_pair(img)

        images_list = []
        for single_pair in img:
            processed = CustomMixedMNISTDataset.process_single_pair(single_pair)
            images_list.append(processed)
        return torch.stack((images_list))

    @staticmethod
    def process_single_pair(pair):
        normal_img = Image.fromarray(pair[0].numpy(), mode="L")
        outlier_img = Image.fromarray(pair[1].numpy(), mode="L")
        normal_img = train_augs(normal_img)
        outlier_img = train_augs(outlier_img)
        return torch.stack((normal_img, outlier_img), dim=0).squeeze(1)
