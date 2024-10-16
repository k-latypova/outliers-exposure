from torch.utils.data import Dataset


class Cifar10Dataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.length = len(data)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        def process_img(item):
            return self.transform(item).reshape(3, 32, 32)

        items = self.data[idx]
        if len(items.shape) == 3:
            return process_img(items)
        else:
            return [process_img(item) for item in items]
