import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.datasets.cifar_datasets import MixedTestingCIFARDataset
from src.networks.cifar_discriminator import discriminator_nn
from src.networks.cifar_double_cnn_generator import CifarGenerator
from src.utils.cifar_models import (
    CIFARModel,
    EvaluationResult,
    TestResult,
    TrainingResult,
    db,
)
from src.utils.utils import seed_everyting



class CIFARScorer:
    def __init__(
        self,
        generator_state,
        discriminator_state,
        model,
        seed,
        device,
        transform,
        num_workers,
    ):
        seed_everyting(seed, device)
        self.model = model
        self.device = device

        self.discriminator_nn = discriminator_nn.to(device)
        self.generator_nn = CifarGenerator(
            int((model.latent_dim_size - 1) / 2), model.latent_dim_size
        ).to(device)
        self.generator_nn.load_state_dict(generator_state)
        self.discriminator_nn.load_state_dict(discriminator_state)
        self.test_label = -1
        self.transform = transform
        self.num_workers = num_workers
        # self.reverse_transform = reverse_transform

    def prepare_datasets(self, test_label):
        cifar_dataset = datasets.CIFAR10(
            root="./cifar10", download=True, train=False
        )
        normal_labels = list(map(int, self.model.normal_labels.split(",")))
        targets = torch.Tensor(cifar_dataset.targets)
        mask = targets == normal_labels[0]
        for i in range(1, len(normal_labels)):
            mask = mask | (targets == normal_labels[i])
        normal_data = cifar_dataset.data[mask]

        mask = targets == test_label
        outliers_data = cifar_dataset.data[mask]

        self.dataset = MixedTestingCIFARDataset(
            normal_data, outliers_data, transform=self.transform
        )
        if self.device == "cuda":
            pin_memory = True
        else:
            pin_memory = False
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.model.batch_size,
            shuffle=True,
            pin_memory_device=self.device,
            pin_memory=pin_memory,
            num_workers=self.num_workers,
        )

    def calculate_f1_scores(self, predictions, ground_truth, thresholds):
        f1_scores = []
        for threshold in thresholds:
            thresholded_predictions = self.apply_threshold(predictions, threshold)
            f1_score_val = f1_score(ground_truth, thresholded_predictions)
            f1_scores.append(f1_score_val)

        return f1_scores

    def apply_threshold(self, probabilities, threshold):
        mask = probabilities >= threshold
        labels = mask.int()
        return labels

    def score(self, test_label: int):
        self.test_label = test_label
        print(f"Starting evaluating on outliers with label: {test_label}\n", flush=True)
        self.prepare_datasets(test_label)
        self.generator_nn.eval()
        self.discriminator_nn.eval()
        predictions = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([])
        with torch.no_grad():
            for data, labels in self.dataloader:
                data = data.to(self.device)
                prediction = discriminator_nn(data)
                predictions = torch.cat((predictions, prediction))
                ground_truth = torch.cat((ground_truth, labels))

        predictions = predictions.cpu().view_as(ground_truth)
        predictions = predictions.cpu().view_as(ground_truth)
        precision, recall, thresholds = precision_recall_curve(
            ground_truth, predictions
        )

        f1_scores = self.calculate_f1_scores(predictions, ground_truth, thresholds)
        f1_score_value = max(f1_scores)

        return f1_score_value
