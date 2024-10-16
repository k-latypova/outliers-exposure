import os
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (f1_score,
                             precision_recall_curve, roc_auc_score)
from torch import nn
from torch.distributions import Uniform
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.datasets.dataset_constructor import construct_datasets
from src.networks.mnist_discriminator import discriminator_nn
from src.networks.mnist_generator import generator_nn

# invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
#                                                      std = [ 1/0.229, 1/0.224, 1/0.225 ]),
#                                 transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
#                                                      std = [ 1., 1., 1. ]),
#                                ])


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


def disc_loss(disc_pred, ground_truth_val):
    criterion = nn.MSELoss()
    ground_truth = ground_truth_val.view(disc_pred.shape)
    loss = criterion(disc_pred, ground_truth)
    return loss


def gen_loss(gen_point, ground_truth):
    criterion = nn.MSELoss(reduction="sum")
    ground_truth = ground_truth.view(gen_point.shape)
    loss = criterion(gen_point.float(), ground_truth.float())
    return loss


class MixupTrainer:
    def __init__(
        self,
        epochs,
        batch_size,
        outliers_dataset_size,
        lambda_1,
        lambda_2,
        generator_lr,
        discriminator_lr,
        interpolation_sample_size,
        normal_dataset_size,
        device,
        normal_label=1,
        outlier_label=2,
        test_label=3,
        save_results=False,
    ):
        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.outliers_dataset_size = outliers_dataset_size
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.interpolation_sample_size = interpolation_sample_size
        self.normal_dataset_size = normal_dataset_size
        self.normal_label = normal_label
        self.outliers_label = outlier_label
        if save_results:
            self.__create_output_directory()

        self.__generator_nn = generator_nn.to(self.device)
        self.__discriminator_nn = discriminator_nn.to(self.device)
        # self.dataset = CustomMixedMNISTDataset(self.normal_dataset_size, self.outliers_dataset_size, 1, 2)
        training_dataset, test_dataset, test_label_dataset = construct_datasets(
            self.normal_dataset_size,
            self.outliers_dataset_size,
            normal_label,
            outlier_label,
            test_label,
            [0.8, 0.2],
        )
        self.train_dataset = training_dataset
        self.test_dataset = test_dataset
        self.test_label_dataset = test_label_dataset
        self.interpolation_distribution = Uniform(0, 1)
        self.gen_train_losses = []
        self.d_train_losses = []
        self.save_results = save_results

    def __split(self, lengths: List):
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            self.dataset, lengths
        )

    def __create_output_directory(self):
        current_date = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        path = os.path.join("out", current_date)
        Path(path).mkdir(exist_ok=True, parents=True)
        params_filename = os.path.join(path, "params.txt")
        with open(params_filename, "w") as params_file:
            params_file.write(
                f"Size of normal dataset: {self.normal_dataset_size}\n"
                f"Number of outliers: {self.outliers_dataset_size}\n"
                f"Interpolation sample size: {self.interpolation_sample_size}\n"
                f"lambda_1: {self.lambda_1}\n"
                f"lambda_2: {self.lambda_2}\n"
                f"Epochs: {self.epochs}\n"
                f"Normal label: {self.normal_label}\n"
                f"Outlier label: {self.outliers_label}\n"
            )
        self.__output_path = path
        self.__log_file_name = os.path.join(path, "logs.txt")

    def train(self, num_workers=1):
        self.__discriminator_nn.apply(weights_init)
        self.__generator_nn.apply(weights_init)
        discriminator_optimizer = torch.optim.Adam(
            self.__discriminator_nn.parameters(), lr=self.discriminator_lr
        )
        generator_optimizer = torch.optim.Adam(
            self.__generator_nn.parameters(), lr=self.generator_lr
        )
        discriminator_scheduler = MultiStepLR(
            discriminator_optimizer, milestones=[50, 130]
        )
        generator_scheduler = MultiStepLR(generator_optimizer, milestones=[80])

        trainloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        avg_disc_loss = 0.0
        avg_gen_loss = 0.0
        for i in tqdm(range(1, self.epochs + 1)):
            total_discriminator_loss = 0.0
            total_generator_loss = 0.0
            for idx, data in tqdm(enumerate(trainloader, 1), leave=False):
                data = data.to(self.device)
                discriminator_optimizer.zero_grad()
                convoluted = self.__generator_nn.conv_layers(data)
                interpolations = self.interpolation_distribution.sample(
                    (self.interpolation_sample_size, 1)
                ).to(self.device)
                interpolations_batch = interpolations.repeat(1, data.size(0)).view(
                    -1, 1
                )
                vectors_batch = convoluted.repeat(self.interpolation_sample_size, 1)
                gen_input = (
                    torch.cat((vectors_batch, interpolations_batch), 1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                generated_outliers = self.__generator_nn.generator_layers(gen_input)
                generated_outliers = generated_outliers.view(-1, 1, 28, 28)
                predicted_scores = self.__discriminator_nn(generated_outliers)
                disc_batch_loss = disc_loss(predicted_scores, interpolations_batch)
                # disc_batch_loss = 0.0
                # for interpolation in interpolations:
                #     interpolation_batch = torch.ones(data.size(0), 1, device=self.device) * interpolation
                #     generated_outlier = self.__generator_nn(data, interpolation_batch).to(self.device)
                #     regressed_interpolation = self.__discriminator_nn(generated_outlier).to(self.device)
                #     disc_ineterp_loss = disc_loss(regressed_interpolation, interpolation_batch)
                #     disc_batch_loss += disc_ineterp_loss
                #     del regressed_interpolation
                #     del interpolation_batch

                # disc_batch_loss /= self.interpolation_sample_size
                # del interpolations
                disc_batch_loss.backward(retain_graph=True)
                discriminator_optimizer.step()

                if idx % 5 == 0:
                    generator_optimizer.zero_grad()
                    gen_normal_input = (
                        torch.cat(
                            (
                                convoluted,
                                torch.zeros(convoluted.size(0), 1, device=self.device),
                            ),
                            1,
                        )
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    gen_outlier_input = (
                        torch.cat(
                            (
                                convoluted,
                                torch.ones(convoluted.size(0), 1, device=self.device),
                            ),
                            1,
                        )
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                    generated_total_outlier = (
                        self.__generator_nn.generator_layers(gen_outlier_input)
                        .view(-1, 1, 28, 28)
                        .to(self.device)
                    )
                    generated_total_normal = (
                        self.__generator_nn.generator_layers(gen_normal_input)
                        .view(-1, 1, 28, 28)
                        .to(self.device)
                    )
                    gen_loss_outlier_value = gen_loss(
                        generated_total_outlier, data[:, 1, :, :]
                    )
                    gen_loss_normal_value = gen_loss(
                        generated_total_normal, data[:, 0, :, :]
                    )
                    del generated_total_outlier
                    del generated_total_normal
                    gen_loss_value = (
                        gen_loss_normal_value * self.lambda_1
                        + gen_loss_outlier_value * self.lambda_2
                    )
                    gen_loss_value.backward()
                    generator_optimizer.step()
                    total_generator_loss += gen_loss_value.item()
                total_discriminator_loss += disc_batch_loss.item()

            avg_disc_loss = total_discriminator_loss / len(trainloader)
            avg_gen_loss = total_generator_loss / (len(trainloader))
            self.gen_train_losses.append(avg_gen_loss)
            self.d_train_losses.append(avg_disc_loss)

            discriminator_scheduler.step()
            generator_scheduler.step()

            if i % 10 == 0 and self.save_results:
                log_message = (f"Epoch: {i} | D: loss {avg_disc_loss} | G: loss {avg_gen_loss} | "
                               f"lr: {discriminator_scheduler.get_last_lr()}\n")
                self.__log(log_message)
                self.__save_images(i, avg_disc_loss, avg_gen_loss)

    def __log(self, log_message):
        print(log_message)
        if self.save_results:
            with open(self.__log_file_name, "a") as log_file:
                log_file.write(log_message)

    def __save_images(self, epoch, d_error, g_error):
        with torch.no_grad():
            self.__generator_nn.eval()
            self.__discriminator_nn.eval()
            interpolations = torch.arange(0, 1.1, step=0.1, device=self.device)
            test_size = 10

            generated_outliers = []
            expected_outliers = []
            loader = DataLoader(self.train_dataset, batch_size=test_size, shuffle=True)
            data = next(iter(loader)).to(self.device)
            _, axs = plt.subplots(10, 13, figsize=(28, 28))
            axs = axs.flatten()

            for pair, ax in zip(data, axs):
                generated_outliers.append((pair[0, :, :], "normal"))
                sanity_check_outliers = [(None, "")]
                for interpolation in interpolations:
                    generated_outlier = generator_nn(pair, interpolation.squeeze(0)).to(
                        self.device
                    )
                    sanity_check_outlier = (1 - interpolation) * pair[
                        1, :, :
                    ] + interpolation * pair[0, :, :]
                    sanity_check_outliers.append(
                        (sanity_check_outlier, f"Sanity check, gamma: {interpolation}")
                    )
                    expected_outlier = (
                        interpolation * pair[1, :, :]
                        + (1 - interpolation) * pair[0, :, :]
                    )
                    expected_outliers.append(expected_outlier)
                    prediction = discriminator_nn(generated_outlier)
                    generated_outliers.append(
                        (
                            generated_outlier,
                            f"Gamma: {interpolation: .3f}, Pred: {float(prediction): .3f}",
                        )
                    )
                generated_outliers.append((pair[1, :, :], "outlier"))
                sanity_check_outliers.append((None, ""))
                generated_outliers.extend(sanity_check_outliers)
            del interpolations

            for (img, title), ax in zip(generated_outliers, axs):
                if img is not None:
                    # img = invTrans(img)
                    img_arr = img.detach().cpu().numpy().reshape(28, 28)
                    ax.imshow(img_arr)
                    ax.set_title(title, fontsize=6)
                else:
                    ax.axis("off")
            plt.suptitle(f"D error: {d_error}, G error {g_error}")

            self.__generator_nn.train()
            self.__discriminator_nn.train()
            plt.savefig(os.path.join(self.__output_path, "epoch_{}.pdf".format(epoch)))
            plt.close()

    def test(self):
        return self.test_on_test_label(self.test_dataset, "digit")
        # results = []
        # for digit, dataset in self.test_datasets:
        #     score = self.test_on_test_label(dataset, f"digit{digit}")
        #     results.append((digit, score))
        # return results

    def test_on_test_label(self, test_dataset, dataset_name):
        self.__generator_nn.eval()
        self.__discriminator_nn.eval()
        testloader = DataLoader(test_dataset, batch_size=self.batch_size)
        predictions = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([])
        with torch.no_grad():
            for data, labels in testloader:
                data = data.to(self.device)
                prediction = discriminator_nn(data)
                predictions = torch.cat((predictions, prediction))
                ground_truth = torch.cat((ground_truth, labels))

        predictions = predictions.cpu().view_as(ground_truth)
        precision, recall, thresholds = precision_recall_curve(
            ground_truth, predictions
        )
        f1_scores = np.divide(
            2 * np.multiply(precision, recall),
            np.add(precision, recall),
            out=np.zeros_like(precision),
            where=np.add(precision, recall) != 0,
        )
        f1_score = np.max(f1_scores)

        # f1_scores = self.calculate_f1_scores(predictions, ground_truth, thresholds)

        # auc = roc_auc_score(ground_truth, predictions)
        # self.__log(f"Area under curve for {dataset_name}: {auc}\n")
        # fpr, tpr, _ = roc_curve(ground_truth, predictions)
        #
        # average_prediction_val = average_precision_score(ground_truth, predictions)
        # self.__log(f"Average prediction score for {dataset_name}: {average_prediction_val}\n")
        #
        # fig, axs = plt.subplots(3, 1)
        # axs[0].plot(recall, precision)
        # axs[0].set(xlabel='Recall', ylabel='Precision')
        # axs[0].set_title('Precision recall curve')
        # axs[1].plot(thresholds, f1_scores, 'tab:orange')
        # axs[1].set(xlabel='Thresholds', ylabel='F1 scores')
        # axs[1].set_title('F1 scores')
        # axs[2].plot(fpr, tpr, 'tab:green')
        # axs[2].set_title('ROC curve')
        #
        # plt.subplots_adjust(top=1.0, hspace=0.6)
        # file_path = os.path.join(self.__output_path, f'Test metrics curves for dataset {dataset_name}.pdf')
        # plt.savefig(file_path)
        # print(file_path)
        # plt.close()
        return f1_score

    # def validate_model(self):
    #     self.__generator_nn.eval()
    #     self.__discriminator_nn.eval()
    #     val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
    #     predictions = torch.tensor([], device=self.device)
    #     ground_truth = torch.tensor([])
    #     with torch.no_grad():
    #         for data, labels in val_loader:
    #             data = data.to(self.device)
    #             prediction = discriminator_nn(data)
    #             predictions = torch.cat((predictions, prediction))
    #             ground_truth = torch.cat((ground_truth, labels))
    #
    #     predictions = predictions.cpu().view_as(ground_truth)
    #     discriminator_loss = disc_loss(predictions, ground_truth)

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

    def plot_losses(self):
        with torch.no_grad():
            plt.plot(
                np.linspace(1, self.epochs, self.epochs).astype(int),
                self.gen_train_losses,
            )
            file_path = os.path.join(self.__output_path, "generator_loss_curve.png")
            plt.savefig(file_path)
            plt.clf()
            plt.plot(
                np.linspace(1, self.epochs, self.epochs).astype(int),
                self.d_train_losses,
            )
            file_path = os.path.join(
                self.__output_path, "discriminator_loss_curve.png"
            )
            plt.savefig(file_path)
            plt.close()

    def score(self):
        self.__generator_nn.eval()
        self.__discriminator_nn.eval()
        testloader = DataLoader(self.test_label_dataset, batch_size=self.batch_size)
        predictions = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([])
        with torch.no_grad():
            for data, labels in testloader:
                data = data.to(self.device)
                prediction = discriminator_nn(data)
                predictions = torch.cat((predictions, prediction))
                ground_truth = torch.cat((ground_truth, labels))

        predictions = predictions.cpu().view_as(ground_truth)

        try:
            auc = roc_auc_score(ground_truth, predictions)
        except Exception as e:
            print(f"predictions: {predictions}\n")
            print(f"ground truth: {ground_truth}\n")
            print(
                (f"g_lr: {self.generator_lr}, d_lr: {self.discriminator_lr}, "
                 f"lambda 1: {self.lambda_1}, lambda_2: {self.lambda_2}\n")
            )
            raise e

        return auc
