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


class CIFAREvaluator:
    def __init__(
        self,
        model: CIFARModel,
        device,
        transform,
        seed=42,
        epoch=None,
        training_result: TrainingResult = None,
        num_outliers: int = 1000,
        num_seeds: int = 5,
    ):
        seed_everyting(seed, device)
        self.model = model
        self.device = device
        state_path = model.weights_path
        self.training_result = training_result
        self.num_seeds = num_seeds
        self.num_outliers = num_outliers
        if epoch and not training_result:
            with db:
                training_result = TrainingResult.get(
                    (TrainingResult.model == model)
                    & (TrainingResult.epoch == epoch)
                    & (TrainingResult.seed == seed)
                )
            state_path = training_result.weights_path
        elif training_result:
            state_path = training_result.weights_path
        generator = CifarGenerator(
            int((model.latent_dim_size - 1) / 2), model.latent_dim_size
        ).to(device)
        self.discriminator_nn = discriminator_nn.to(device)
        states = torch.load(state_path, map_location=torch.device(device))
        self.generator_nn = generator

        self.generator_nn.load_state_dict(states["generator"])
        self.discriminator_nn.load_state_dict(states["discriminator"])
        self.__dir_output = os.path.dirname(state_path)
        print(
            f"Metrics.txt will be saved to  are saved: {self.__dir_output}", flush=True
        )
        self.test_labels = None
        self.means = list(map(float, self.model.means.split(",")))
        self.stds = list(map(float, self.model.stds.split(",")))
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.means, self.stds)]
        )

    def __log(self, msg):
        log_file_path = os.path.join(
            self.__dir_output, f"metrics_{self.test_labels}.txt"
        )
        with open(log_file_path, "a") as log_file:
            print(msg + "\n")
            log_file.write(msg)

    def prepare_datasets(self, test_labels):
        # cifar_dataset = datasets.CIFAR10(root='./cifar10', download=True, train=False)
        cifar_dataset = datasets.CIFAR10(
            root="./cifar10", download=True, train=False
        )
        normal_labels = list(map(int, self.model.normal_labels.split(",")))
        targets = torch.Tensor(cifar_dataset.targets)
        mask = targets == normal_labels[0]
        for i in range(1, len(normal_labels)):
            mask = mask | (targets == normal_labels[i])
        normal_data = cifar_dataset.data[mask]
        self.normal_data = normal_data

        # mask = targets == test_labels[0]
        self.datasets = []
        for label in test_labels:
            mask = targets == label
            outliers_data = cifar_dataset.data[mask]

            # dataset = MixedTestingCIFARDataset(normal_data, outliers_data, transform=self.transform)
            self.datasets.append((label, outliers_data))

    def evaluate(self, test_labels, num_workers=1):
        self.test_labels = test_labels
        self.__log(f"Starting evaluating on outliers with labels: {test_labels}\n")
        self.prepare_datasets(test_labels)
        self.generator_nn.eval()
        self.discriminator_nn.eval()
        test_results = []

        for label, outliers_data in self.datasets:
            dataset = MixedTestingCIFARDataset(
                normal_data=self.normal_data,
                outliers_data=outliers_data,
                transform=self.transform,
            )
            seeds = np.random.choice(10000, self.num_seeds, replace=False)
            for seed in seeds:
                seed_everyting(seed, self.device)
            testloader = DataLoader(
                dataset,
                batch_size=self.model.batch_size,
                num_workers=num_workers,
                shuffle=False,
            )
            predictions = torch.tensor([], device=self.device)
            ground_truth = torch.tensor([])
            with torch.no_grad():
                for data, labels in testloader:
                    data = data.to(self.device)
                    prediction = self.discriminator_nn(data)
                    predictions = torch.cat((predictions, prediction))
                    ground_truth = torch.cat((ground_truth, labels))

            predictions = predictions.cpu().view_as(ground_truth)
            precision, recall, thresholds = precision_recall_curve(
                ground_truth, predictions
            )

            f1_scores = self.calculate_f1_scores(predictions, ground_truth, thresholds)

            auc = roc_auc_score(ground_truth, predictions)
            self.__log(f"Evaluating for label: {label}\n")
            self.__log(f"AUC: {auc:.3f}\n")
            fpr, tpr, _ = roc_curve(ground_truth, predictions)

            average_prediction_val = average_precision_score(ground_truth, predictions)
            self.__log(f"Average prediction score: {average_prediction_val:.3f}\n")

            f1_score = max(f1_scores)
            self.__log(
                f"Max f1 score: {f1_score:.3f} for threshold: {thresholds[f1_scores.index(f1_score)]}\n"
            )

            fig, axs = plt.subplots(3, 1)
            axs[0].plot(recall, precision)
            axs[0].set(xlabel="Recall", ylabel="Precision")
            axs[0].set_title("Precision recall curve")
            axs[1].plot(thresholds, f1_scores, "tab:orange")
            axs[1].set(xlabel="Thresholds", ylabel="F1 scores")
            axs[1].set_title("F1 scores")
            axs[2].plot(fpr, tpr, "tab:green")
            axs[2].set_title("ROC curve")

            plt.subplots_adjust(top=1.0, hspace=0.6)
            test_result = TestResult(
                model=self.model.id,
                average_prediction_score=average_prediction_val,
                auc=auc,
                f1_score=max(f1_scores),
                test_label=label,
                seed=-1,
            )
            test_results.append(test_result)

            plt.savefig(os.path.join(self.__dir_output, f"metrics_{label}.pdf"))
            plt.close()

        with db:
            for test_result in test_results:
                test_result.save()

            evaluation_result = EvaluationResult(
                model=self.model.id,
                epoch=self.training_result.epoch if self.training_result else -1,
            )
            evaluation_result.auc = sum(ts.auc for ts in test_results) / len(
                test_results
            )
            evaluation_result.f1_score = sum(ts.f1_score for ts in test_results) / len(
                test_results
            )
            evaluation_result.average_prediction_score = sum(
                ts.average_prediction_score for ts in test_results
            ) / len(test_results)

            evaluation_result.save()
        return test_results

    def run(self, test_labels):
        self.prepare_normal_data()
        self.run_normal_data()

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
