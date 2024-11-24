from typing import List

import torch
from torchvision import datasets, transforms as T
from torch.utils.data import random_split
from datasets.mixed_training_dataset import MixedTrainingMNISTDataset, MixedTestingMNISTDataset
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



