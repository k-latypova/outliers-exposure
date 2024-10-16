import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.datasets.cifar10_dataset import Cifar10Dataset
from src.datasets.cifar_datasets import MixedTestingCIFARDataset
from src.networks.cifar_discriminator import CIFARDiscriminator
from src.utils.cifar_models import EvalResult, db
from src.utils.utils import calculate_f1_scores, seed_everyting


class CifarEvaluator:
    def __init__(self, models, device):
        self.models = models
        self.normal_labels = models[0][0].normal_labels
        self.outliers_labels = models[0][0].outliers_labels
        self.train_outliers_num = models[0][0].outliers_num
        self.normal_data = None
        self.outlier_data = None
        self.device = device
        self.discriminator_nn = CIFARDiscriminator().to(device)
        self.__dir_output = os.path.join(
            os.sep,
            "scratch",
            "latypova",
            "oe",
            "evaluation results",
            f"{self.normal_labels}-{self.outliers_labels}-{self.train_outliers_num}",
        )
        Path(self.__dir_output).mkdir(exist_ok=True, parents=True)
        self.outliers_nums = [len(self.outliers_labels.split(",")) * 1000]
        self.pin_memory = True if device == "cuda" else False
        self.metrics = []
        self.transform = None
        self.batch_size = 32
        self.num_workers = 1

    def init_networks_weights(self, training_result):
        state_path = training_result.weights_path
        states = torch.load(state_path, map_location=torch.device(self.device))
        self.discriminator_nn = CIFARDiscriminator().to(self.device)
        self.discriminator_nn.load_state_dict(states["discriminator"])

    def init_transforms(self, model):
        self.means = list(map(float, model.means.split(",")))
        self.stds = list(map(float, model.stds.split(",")))
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(self.means, self.stds)]
        )
        self.normal_dataset.transform = self.transform
        self.outlier_dataset.transform = self.transform

    def prepare_normal_data(self):
        cifar_dataset = datasets.CIFAR10(
            root="./cifar10", download=True, train=False
        )
        normal_labels = list(map(int, self.normal_labels.split(",")))
        targets = torch.Tensor(cifar_dataset.targets)
        mask = targets == normal_labels[0]
        for i in range(1, len(normal_labels)):
            mask = mask | (targets == normal_labels[i])
        self.normal_data = cifar_dataset.data[mask]
        self.normal_dataset = Cifar10Dataset(self.normal_data, self.transform)

    def prepare_outlier_data(self, test_label: int):
        cifar_dataset = datasets.CIFAR10(
            root="./cifar10", download=True, train=False
        )
        targets = torch.Tensor(cifar_dataset.targets)
        mask = targets == test_label
        self.outlier_data = cifar_dataset.data[mask]
        self.outlier_dataset = Cifar10Dataset(self.outlier_data, self.transform)

    def run_predictions(self, batch_size, num_workers, dataset):
        self.discriminator_nn.eval()
        predictions = torch.tensor([], device=self.device)
        with torch.no_grad():
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=self.pin_memory,
                pin_memory_device=self.device,
            )
            for data in loader:
                data = data.to(self.device)
                batch_pred = self.discriminator_nn(data)
                predictions = torch.cat((predictions, batch_pred), dim=0)
        return predictions

    def evaluate_results(self, predictions, ground_truth, outliers_num):
        precision, recall, thresholds = precision_recall_curve(
            ground_truth, predictions
        )
        f1_scores = calculate_f1_scores(predictions, ground_truth, thresholds)
        optimal_f1_score = np.max(f1_scores)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        fpr, tpr, _ = roc_curve(ground_truth, predictions)

        # calculate f1 score for threshold=0.5
        thresholded_predictions = predictions > 0.5
        f1_score_val = f1_score(ground_truth, thresholded_predictions)
        self.__log(f"F1 score for threshold=0.5: {f1_score_val:.3f}\n")

        # calculate f1 score for threshold=0.2
        thresholded_predictions = predictions > 0.15
        f1_score_val = f1_score(ground_truth, thresholded_predictions)
        self.__log(f"F1 score for threshold=0.2: {f1_score_val:.3f}\n")

        average_prediction_val = average_precision_score(ground_truth, predictions)
        #  self.__log(f"Average prediction score: {average_prediction_val:.3f}\n")

        auc = roc_auc_score(ground_truth, predictions)
        # auprc = roc_auc_score(recall, precision)
        auprc = average_prediction_val

        # self.__log(f"AUC: {auc*100:.3f}%\n")
        #
        self.__log(f"Optimal threshold: {optimal_threshold:.3f}\n")
        #
        # self.__log(f"Optimal F1 score: {optimal_f1_score:.3f}\n")

        metric = {
            "outliers_num": outliers_num,
            "pr_curve": (precision, recall, thresholds),
            "roc_curve": (fpr, tpr),
            "optimal_f1_score": optimal_f1_score,
            "optimal_threshold": optimal_threshold,
            "f1_scores": (f1_scores, thresholds),
            "average_prediction_val": average_prediction_val,
            "auc": auc,
            "auprc": auprc,
        }
        return metric

    def build_plots(self, test_label):
        self.__build_f1_scores(test_label)
        self.__build_auc(test_label)
        self.__build_average_prediction_val(test_label)

    # def __build_pr_curve(self, metric, test_label):
    #     precision, recall, thresholds = metric["pr_curve"]
    #     plt.plot(recall, precision, marker='.', label='PR Curve')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.legend()
    #     plt.savefig(os.path.join(self.__dir_output, f"pr_curve_{metric['outliers_num']}_{test_label}.png"))
    #     plt.clf()
    #
    # def __build_roc_curve(self, metric, test_label):
    #     fpr, tpr = metric["roc_curve"]
    #     plt.plot(fpr, tpr, marker='.', label='ROC Curve')
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.legend()
    #     plt.savefig(os.path.join(self.__dir_output, f"roc_curve_{metric['outliers_num']}_{test_label}.png"))
    #     plt.clf()

    def __build_f1_scores(self, test_label):
        outliers_num = self.outliers_nums
        f1_scores = [m["f1_scores"] for _, m in self.metrics.items()]
        plt.boxplot(f1_scores, labels=outliers_num)
        plt.xlabel("Outliers num")
        plt.ylabel("F1 score")
        plt.savefig(os.path.join(self.__dir_output, f"f1_scores_{test_label}.png"))
        path = os.path.join(self.__dir_output, f"f1_scores_{test_label}.png")
        print(f"Saved the plot at {path}")
        plt.close()

    def __build_average_prediction_val(self, test_label):
        outliers_num = self.outliers_nums
        average_prediction_val = [
            m["average_prediction_vals"] for _, m in self.metrics.items()
        ]
        plt.boxplot(average_prediction_val, labels=outliers_num)
        plt.xlabel("Outliers num")
        plt.ylabel("Average prediction value")
        plt.savefig(
            os.path.join(self.__dir_output, f"average_prediction_val_{test_label}.png")
        )
        plt.close()

    def __build_auc(self, test_label):
        outliers_num = self.outliers_nums
        auc = [m["auc_scores"] for _, m in self.metrics.items()]
        plt.boxplot(auc, labels=outliers_num)
        plt.xlabel("Outliers num")
        plt.ylabel("AUC")
        plt.savefig(os.path.join(self.__dir_output, f"auc_{test_label}.png"))
        plt.close()

    def __log(self, message):
        print(message)
        with open(os.path.join(self.__dir_output, "log.txt"), "a") as f:
            f.write(message)
            f.write("\n")

    def get_mixed_predictions(self, dataset):
        self.discriminator_nn.eval()
        with torch.no_grad():
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            predictions = torch.tensor([]).to(self.device)
            ground_truth = torch.tensor([])
            for data, labels in loader:
                data = data.to(self.device)
                batch_pred = self.discriminator_nn(data)
                predictions = torch.cat((predictions, batch_pred), dim=0)
                ground_truth = torch.cat((ground_truth, labels), dim=0)
            return predictions, ground_truth

    def run(self, test_label, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prepare_normal_data()
        self.prepare_outlier_data(test_label)

        seeds = np.random.choice(10000, 5, replace=False)
        self.metrics = {
            x: {
                "f1_scores": [],
                "auc_scores": [],
                "average_prediction_vals": [],
                "auprc": [],
            }
            for x in self.outliers_nums
        }

        for model, training_result in self.models:
            self.init_networks_weights(training_result)
            self.init_transforms(model)
            self.discriminator_nn.eval()
            # normal_predictions = self.run_predictions(batch_size, num_workers, self.normal_dataset)
            # outliers_predictions = self.run_predictions(batch_size, num_workers, self.outlier_dataset)

            for outliers_num in self.outliers_nums:
                for seed in seeds:
                    seed_everyting(seed, self.device)
                    index = np.random.choice(
                        self.outlier_data.shape[0], outliers_num, replace=False
                    )
                    dataset = MixedTestingCIFARDataset(
                        self.normal_data, self.outlier_data[index], self.transform
                    )
                    predictions, ground_truth = self.get_mixed_predictions(dataset)
                    # cur_outliers_predictions = outliers_predictions[index]
                    # predictions = torch.cat((normal_predictions, cur_outliers_predictions), dim=0)
                    # ground_truth = torch.cat((torch.zeros(normal_predictions.shape[0], device=self.device),
                    #                         torch.ones(cur_outliers_predictions.shape[0], device=self.device)), dim=0)
                    metric = self.evaluate_results(
                        predictions.detach().cpu(),
                        ground_truth.detach().cpu(),
                        outliers_num,
                    )
                    self.metrics[outliers_num]["f1_scores"].append(
                        metric["optimal_f1_score"]
                    )
                    self.metrics[outliers_num]["auc_scores"].append(metric["auc"])
                    self.metrics[outliers_num]["average_prediction_vals"].append(
                        metric["average_prediction_val"]
                    )
                    self.metrics[outliers_num]["auprc"].append(metric["auprc"])
                # self.metrics.append(self.average_metrics(metrics, outliers_num))

        self.build_plots(test_label)
        self.save_results(test_label)

    def save_results(self, test_label):
        with db:
            for outliers_num, metric in self.metrics.items():
                average_f1_score = np.mean(metric["f1_scores"])
                average_prediction_val = np.mean(metric["average_prediction_vals"])
                average_auc = np.mean(metric["auc_scores"])
                eval_result = EvalResult(
                    test_label=test_label,
                    model_outliers_num=self.train_outliers_num,
                    test_outliers_num=outliers_num,
                    f1_score=average_f1_score,
                    average_prediction_score=average_prediction_val,
                    auc_score=average_auc,
                    f1_scores=",".join([str(x) for x in metric["f1_scores"]]),
                    created_at=datetime.now(),
                    normal_labels=self.normal_labels,
                    model_ids=",".join([str(x.id) for x, _ in self.models]),
                    auprc=np.mean(metric["auprc"]),
                    outliers_labels=self.outliers_labels,
                )
                eval_result.save()

    def average_metrics(self, metrics, outliers_num):
        average_metric = {
            "outliers_num": outliers_num,
            "optimal_f1_scores": [m["optimal_f1_score"] for m in metrics],
            "average_prediction_val": [m["average_prediction_val"] for m in metrics],
            "auc": [m["auc"] for m in metrics],
        }
        return average_metric

    def average_roc_curve(self, metrics):
        fpr = np.mean([metric["roc_curve"][0] for metric in metrics], axis=0)
        tpr = np.mean([metric["roc_curve"][1] for metric in metrics], axis=0)
        return fpr, tpr

    def average_pr_curve(self, metrics):
        precision = np.mean([metric["pr_curve"][0] for metric in metrics], axis=0)
        recall = np.mean([metric["pr_curve"][1] for metric in metrics], axis=0)
        thresholds = np.mean([metric["pr_curve"][2] for metric in metrics], axis=0)
        return precision, recall, thresholds

    def average_optimal_f1_score(self, metrics):
        return np.mean([metric["optimal_f1_score"] for metric in metrics])

    def average_optimal_threshold(self, metrics):
        return np.mean([metric["optimal_threshold"] for metric in metrics])

    def average_f1_scores(self, metrics):
        f1_scores = np.mean([metric["f1_scores"][0] for metric in metrics], axis=0)
        thresholds = np.mean([metric["f1_scores"][1] for metric in metrics], axis=0)
        return f1_scores, thresholds
