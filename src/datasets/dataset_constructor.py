from typing import List

import torch
from torchvision import datasets, transforms as T
from torch.utils.data import random_split, Subset
from datasets.mixed_training_dataset import MixedTrainingMNISTDataset, MixedTestingMNISTDataset
from datasets.cifar_datasets import MixedTrainingCIFARDataset, MixedTestingCIFARDataset
import numpy as np

train_augs = T.Compose([T.RandomRotation(20), T.ToTensor()])

mnist_dataset = datasets.MNIST('MNIST/', download=True, train=True, transform=train_augs)
cifar_dataset = datasets.CIFAR10(root='./cifar10', download=True, train=True)

def construct_datasets(normal_sample_size, outlier_sample_size, normal_labels, outlier_label, test_label,
                       split_lengths: List[float]):

    # retrieve normal data
    mask = np.array(mnist_dataset.targets) == normal_labels[0]


    normal_label_data = mnist_dataset.data[mask]
    normal_sample_size = min(normal_sample_size, len(normal_label_data))
    normal_indices = torch.randperm(len(normal_label_data))
    normal_data = normal_label_data[normal_indices]
    training_normal_data, test_normal_data = random_split(normal_data, split_lengths)

    # retrieve outliers data for training and for same class testing
    mask = mnist_dataset.targets == outlier_label
    outlier_label_data = mnist_dataset.data[mask]
    outliers_sample_size = min(outlier_sample_size, len(outlier_label_data))
    outliers_indices = torch.randperm(len(outlier_label_data))[:outliers_sample_size]
    outliers_data = outlier_label_data[outliers_indices]
    training_outliers, test_outliers = random_split(outliers_data, split_lengths)
    if len(test_outliers) == 0:
        test_outliers = training_outliers

    #retrieve test outliers to test on a different class
    mask = mnist_dataset.targets == test_label
    test_label_outliers_data = mnist_dataset.data[mask]
    test_label_outliers_indices = torch.randperm(len(test_label_outliers_data))[:len(test_normal_data)]
    test_label_outliers_data = test_label_outliers_data[test_label_outliers_indices]
    test_label_dataset = MixedTestingMNISTDataset(test_normal_data.dataset[test_normal_data.indices],
                                                  test_label_outliers_data)

    training_dataset = MixedTrainingMNISTDataset(training_normal_data, training_outliers)
    test_dataset = MixedTestingMNISTDataset(test_normal_data.dataset[test_normal_data.indices], test_outliers.dataset[test_outliers.indices])

    return training_dataset, test_dataset, test_label_dataset


def construct_cifar_datasets(normal_sample_size, outlier_sample_size, normal_labels, outlier_labels, test_label,
                       split_lengths: List[float]):
    targets = torch.Tensor(cifar_dataset.targets)

    # retrieve normal data
    mask = targets == normal_labels[0]
    for i in range(1, len(normal_labels)):
        mask = mask | (targets == normal_labels[i])

    normal_label_data = cifar_dataset.data[mask]

    normal_sample_size = min(normal_sample_size, len(normal_label_data))
    normal_indices = torch.randperm(len(normal_label_data))[:normal_sample_size]
    normal_data = torch.Tensor(normal_label_data[normal_indices])
    training_normal_data, test_normal_data = random_split(normal_data, split_lengths)

    # retrieve outliers data for training and for same class testing
    mask = targets == outlier_labels[0]
    for i in range(1, len(outlier_labels)):
        mask = mask | (targets == outlier_labels[i])
    outlier_label_data = cifar_dataset.data[mask]
    outliers_sample_size = min(outlier_sample_size, len(outlier_label_data))
    outliers_indices = torch.randperm(len(outlier_label_data))[:outliers_sample_size]
    outliers_data = torch.Tensor(outlier_label_data[outliers_indices]).view(-1, 32, 32, 3)
    training_outliers, test_outliers = random_split(outliers_data, split_lengths)
    if len(test_outliers) == 0:
        test_outliers = training_outliers

    # retrieve test outliers to test on a different class
    mask = targets == test_label
    test_label_outliers_data = cifar_dataset.data[mask]
    test_label_outliers_indices = torch.randperm(len(test_label_outliers_data))[:len(test_normal_data)]
    test_label_outliers_data = torch.Tensor(test_label_outliers_data[test_label_outliers_indices])

    training_dataset = MixedTrainingCIFARDataset(training_normal_data, training_outliers)
    test_dataset = MixedTestingCIFARDataset(test_normal_data.dataset[test_normal_data.indices],
                                            test_outliers.dataset[test_outliers.indices])
    test_label_dataset = MixedTestingCIFARDataset(test_normal_data.dataset[test_normal_data.indices],
                                                  test_label_outliers_data)
    torch.cuda.empty_cache()

    return training_dataset, test_dataset, test_label_dataset


def construct_test_cifar_datasets(test_size, normal_labels, test_labels,
                                  split_lengths: List[float], outliers_labels, outliers_sample_size):
    targets = torch.Tensor(cifar_dataset.targets)

    # retrieve normal data
    mask = targets == normal_labels[0]
    for i in range(1, len(normal_labels)):
        mask = mask | (targets == normal_labels[i])

    normal_label_data = cifar_dataset.data[mask]

    normal_sample_size = min(test_size, len(normal_label_data))
    normal_indices = torch.randperm(len(normal_label_data))[:normal_sample_size]
    normal_data = torch.Tensor(normal_label_data[normal_indices])
    training_normal_data, test_normal_data = random_split(normal_data, split_lengths)

    # retrieve outliers data for training and for same class testing
    mask = targets == outliers_labels[0]
    for i in range(1, len(outliers_labels)):
        mask = mask | (targets == outliers_labels[i])
    outlier_label_data = cifar_dataset.data[mask]
    outliers_sample_size = min(outliers_sample_size, len(outlier_label_data))
    outliers_indices = torch.randperm(len(outlier_label_data))[:outliers_sample_size]
    outliers_data = torch.Tensor(outlier_label_data[outliers_indices]).view(-1, 32, 32, 3)
    training_outliers, _ = random_split(outliers_data, split_lengths)

    training_data = torch.cat((normal_data[training_normal_data.indices], outliers_data[training_outliers.indices]))
    mean = training_data.mean(axis=(0, 1, 2)) / 255
    std = training_data.std(axis=(0, 1, 2)) / 255
    transform = T.Compose(
        [T.ToTensor(),
         T.Normalize(mean, std)]
    )



    test_datasets = []
    # retrieve test outliers to test on a different class
    for label in test_labels:
        mask = targets == label
        test_label_outliers_data = cifar_dataset.data[mask]
        test_label_outliers_indices = torch.randperm(len(test_label_outliers_data))[:len(test_normal_data)]
        test_label_outliers_data = torch.Tensor(test_label_outliers_data[test_label_outliers_indices])


        test_label_dataset = MixedTestingCIFARDataset(test_normal_data.dataset[test_normal_data.indices],
                                                  test_label_outliers_data, transform=transform)
        test_datasets.append((label, test_label_dataset))

    # retrieve test outliers to test on a different class

    torch.cuda.empty_cache()

    return test_datasets, dict(transform=transform, mean=mean, std=std)


