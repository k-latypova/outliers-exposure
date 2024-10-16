import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.distributed as dist
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve
from torch import nn
from torch.distributions import Uniform
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, RandomSampler, random_split
from torchvision import datasets
from torchvision import transforms
from torchvision import transforms as T
from tqdm import tqdm

from src.datasets.cifar_datasets import MixedTestingCIFARDataset
from src.datasets.cifar10_dataset import Cifar10Dataset
from src.datasets.dataset_constructor import construct_cifar_datasets
from src.networks.cifar_discriminator import CIFARDiscriminator
from src.networks.cifar_double_cnn_generator import CifarGenerator
from src.utils.cifar_models import CIFARModel, TrainingResult, db
from src.utils.utils import calculate_f1_scores

cifar_dataset = datasets.CIFAR10(root="./cifar10", download=True, train=True)
cifar_test_dataset = datasets.CIFAR10(root="./cifar10", download=True, train=False)


def setup(rank=-1, world_size=-1):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # dist.init_process_group('gloo', rank, )


def cleanup():
    dist.destroy_process_group()


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


def xavier_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight, 0.02)


def disc_loss(disc_pred, ground_truth_val):
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    ground_truth = ground_truth_val.view(disc_pred.shape)
    loss = criterion(disc_pred, ground_truth)
    return loss


def gen_loss(gen_point, ground_truth):
    criterion = nn.MSELoss(reduction="mean")
    ground_truth = ground_truth.view(gen_point.shape)
    loss = criterion(gen_point.float(), ground_truth.float())
    return loss


def gen_loss_1(output, batch):
    """
    Given a batch of images, this function returns the reconstruction loss (MSE in our case)
    """
    try:
        loss = nn.functional.mse_loss(batch, output, reduction="none")
    except RuntimeError as e:
        print(
            f"The generated sample device: {batch.device}, the data devcie: {output.device}"
        )
        raise e
    loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
    return loss


def discriminator_loss(pred, is_real):
    criterion = nn.BCELoss()
    if is_real:
        loss = criterion(pred, torch.ones_like(pred))
    else:
        loss = criterion(pred, torch.zeros_like(pred))
    return loss


def seed_everyting(seed, device):
    torch.manual_seed(seed)
    if device != "cpu":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class CIFARTrainingModel:
    def __init__(self, model: CIFARModel, device: str, save=True, num_workers=1):
        # setup(0, 1)
        self.device = device
        latent_dim_size = model.latent_dim_size
        self.vector_dim = int((latent_dim_size - 1) / 2)
        self.generator_nn = CifarGenerator(self.vector_dim, latent_dim_size).to(device)
        self.anomaly_detector_nn = CIFARDiscriminator().to(device)

        self.model = model

        self.save = save
        if save:
            with db:
                model.save()
        self.__create_output_directory()

        self.interpolation_distribution = Uniform(0, 1)
        # self.interpolation_distribution = Beta(torch.tensor([2]).to(self.device), torch.tensor([0.5]).to(self.device))
        self.gen_train_losses = []
        self.d_train_losses = []
        self.best_validation_result = None
        self.training_results: List[TrainingResult] = []
        self.num_workers = num_workers

    def prepare_datasets(self, test_outliers_label):
        normal_labels = (
            list(map(int, self.model.normal_labels.split(",")))
            if self.model.normal_labels
            else []
        )
        outliers_labels = (
            list(map(int, self.model.outliers_labels.split(",")))
            if self.model.outliers_labels
            else []
        )
        training_dataset, test_dataset, alt_test_dataset = construct_cifar_datasets(
            self.model.normal_data_num,
            self.model.outliers_num,
            normal_labels,
            outliers_labels,
            test_outliers_label,
            [0.8, 0.2],
        )
        mean = training_dataset.mean
        std = training_dataset.std
        reverse_mean = -mean / std
        reverse_std = 1 / std
        self.reverse_transform = transforms.Compose(
            [
                transforms.Normalize(mean=reverse_mean, std=reverse_std),
                transforms.ToPILImage(),
            ]
        )

        self.train_dataset, self.validation_dataset = torch.utils.data.random_split(
            training_dataset, [0.8, 0.2]
        )
        self.test_dataset = test_dataset
        self.validation_loader = DataLoader(
            self.validation_dataset, batch_size=self.model.batch_size, shuffle=True
        )

        self.alt_test_dataset = alt_test_dataset

    def prepare_data(self, normal_data_size_num):
        normal_labels = list(map(int, self.model.normal_labels.split(",")))
        outliers_labels = list(map(int, self.model.outliers_labels.split(",")))
        targets = torch.Tensor(cifar_dataset.targets)
        mask = targets == normal_labels[0]
        for i in range(1, len(normal_labels)):
            mask = mask | (targets == normal_labels[i])

        normal_data = cifar_dataset.data[mask]
        if normal_data_size_num is not None:
            normal_data = normal_data[:normal_data_size_num]

        mask = targets == outliers_labels[0]
        for i in range(1, len(outliers_labels)):
            mask = mask | (targets == outliers_labels[i])
        outliers_data = cifar_dataset.data[mask]

        # pick random elements from outliers data
        index = np.random.choice(
            outliers_data.shape[0], self.model.outliers_num, replace=False
        )
        outliers_data_for_training = outliers_data[index]

        training_data = np.concatenate((normal_data, outliers_data_for_training))
        mean = training_data.mean(axis=(0, 1, 2)) / 255
        std = training_data.std(axis=(0, 1, 2)) / 255
        self.__log(f"Mean: {mean}, std: {std}")
        mean_asstr = np.char.mod("%f", mean)
        self.model.means = ",".join(mean_asstr)
        std_asstr = np.char.mod("%f", std)
        self.model.stds = ",".join(std_asstr)
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(32, padding=4),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        self.transform = transform
        reverse_mean = -mean / std
        reverse_std = 1 / std
        self.reverse_transform = transforms.Compose(
            [
                transforms.Normalize(mean=reverse_mean, std=reverse_std),
                transforms.ToPILImage(),
            ]
        )
        self.normal_dataset = Cifar10Dataset(normal_data, transform)
        self.outliers_dataset = Cifar10Dataset(outliers_data_for_training, transform)
        normal_train, normal_val = random_split(self.normal_dataset, [0.8, 0.2])
        self.normal_train = normal_train
        if self.device == "cuda":
            pin_memory = True
        else:
            pin_memory = False

        self.train_normal_loader = DataLoader(
            self.normal_train,
            batch_size=self.model.batch_size,
            sampler=RandomSampler(
                self.normal_train, replacement=True, num_samples=int(1e10)
            ),
            pin_memory=pin_memory,
            pin_memory_device=self.device,
            num_workers=self.num_workers,
        )
        self.val_normal_loader = DataLoader(
            normal_val,
            batch_size=self.model.batch_size,
            pin_memory_device=self.device,
            pin_memory=pin_memory,
            num_workers=self.num_workers,
            drop_last=True,
        )

        self.outliers_loader = DataLoader(
            self.outliers_dataset,
            batch_size=self.model.batch_size,
            sampler=RandomSampler(
                self.outliers_dataset, replacement=True, num_samples=int(1e10)
            ),
            pin_memory=pin_memory,
            pin_memory_device=self.device,
            num_workers=self.num_workers,
        )

    def prepare_test_data(self):
        normal_labels = list(map(int, self.model.normal_labels.split(",")))
        outliers_labels = list(map(int, self.model.outliers_labels.split(",")))
        targets = torch.Tensor(cifar_test_dataset.targets)
        mask = targets == normal_labels[0]
        for i in range(1, len(normal_labels)):
            mask = mask | (targets == normal_labels[i])

        normal_data = cifar_test_dataset.data[mask]

        mask = targets == outliers_labels[0]
        for i in range(1, len(outliers_labels)):
            mask = mask | (targets == outliers_labels[i])
        outliers_data = cifar_test_dataset.data[mask]
        dataset = MixedTestingCIFARDataset(
            normal_data, outliers_data, transform=self.transform
        )
        pin_memory = True if self.device == "cuda" else False
        self.test_loader = DataLoader(
            dataset,
            batch_size=self.model.batch_size,
            shuffle=True,
            pin_memory_device=self.device,
            pin_memory=pin_memory,
            num_workers=self.num_workers,
        )

    def process_lambdas(self):
        # self.model.lambda_1 = 5.0 / self.model.outliers_num
        # self.model.lambda_2 = 50.0 / len(self.normal_train)
        # self.__log(f"My lambdas: {self.model.lambda_1}, {self.model.lambda_2}")
        with db:
            self.model.save()

    def __create_output_directory(self):
        dir_name = f"cifar_{self.model.id}"
        path = os.path.join(os.sep, "scratch", "latypova", "oe", "results", dir_name)
        Path(path).mkdir(exist_ok=True, parents=True)
        params_filename = os.path.join(path, "params.txt")
        try:
            with open(params_filename, "x") as params_file:
                params_file.write(
                    f"Size of normal dataset: {self.model.normal_data_num}\n"
                    f"Number of outliers: {self.model.outliers_num}\n"
                    f"Interpolation sample size: {self.model.interpolations_sample_size}\n"
                    f"lambda_1: {self.model.lambda_1}\n"
                    f"lambda_2: {self.model.lambda_2}\n"
                    f"generator lr: {self.model.gen_lr}\n"
                    f"discriminator lr: {self.model.disc_lr}\n"
                    f"generator lr milestones: {self.model.gen_lr_milestones}\n"
                    f"discriminator lr milestones: {self.model.disc_lr_milestones}\n"
                    f"batch size: {self.model.batch_size}\n"
                )
        except FileExistsError:
            pass
        self.__output_path = path
        self.__log_file_name = os.path.join(path, "logs.txt")

    def train(
        self,
        epochs,
        seed: int,
        init_epochs,
        n_critic,
        save_checkpoint=None,
        load_from_model=False,
        load_from_epoch=None,
        validation_checkpoint=None,
        num_workers=None,
        trainloader=None,
        validationloader=None,
        init_batch_size=None,
        encoder_name: str = None,
        normal_data_num=None,
    ):
        if init_batch_size is None:
            init_batch_size = self.model.batch_size
        self.seed = seed
        seed_everyting(seed, self.device)
        torch.manual_seed(seed)
        gen_milestones = (
            list(map(int, self.model.gen_lr_milestones.split(",")))
            if self.model.gen_lr_milestones
            else []
        )
        disc_milestones = (
            list(map(int, self.model.disc_lr_milestones.split(",")))
            if self.model.disc_lr_milestones
            else []
        )
        self.prepare_data(normal_data_num)
        self.process_lambdas()
        self.prepare_test_data()
        self.gen_optimizer = None
        self.disc_optimizer = None
        start_epoch = 0
        self.discriminator_optimizer = None
        if not load_from_model:
            self.anomaly_detector_nn.apply(weights_init)
            # self.anomaly_detector_nn.apply(xavier_init)
            self.generator_nn.apply(weights_init)
            self.disc_optimizer = torch.optim.Adam(
                self.anomaly_detector_nn.parameters(),
                lr=self.model.disc_lr,
                betas=(0.5, 0.999),
            )
            self.gen_optimizer = torch.optim.Adam(
                self.generator_nn.parameters(), lr=self.model.gen_lr, betas=(0.5, 0.999)
            )
            # self.discriminator_optimizer = torch.optim.Adam(self.discriminator_nn.parameters(), lr=self.model.disc_lr)

            try:
                if encoder_name is None:
                    encoder_name = f"model_cifar_{self.generator_nn.encoded_img_dim}.pt"

                encoder_weights = torch.load(
                    os.path.join("models_out", encoder_name),
                    map_location=torch.device(self.device),
                )["encoder"]
                self.generator_nn.encoder_1.load_state_dict(encoder_weights)
                # self.generator_nn.encoder_2.load_state_dict(encoder_weights)
            except Exception:
                print(f"Failed to load encoder weights from {encoder_name}", flush=True)
                pass
        else:
            if not load_from_epoch:
                model_path = self.model.weights_path
            else:
                with db:
                    training_result = TrainingResult.get(
                        (TrainingResult.model_id == self.model.id)
                        & (TrainingResult.epoch == load_from_epoch)
                        & (TrainingResult.seed == seed)
                    )
                model_path = training_result.weights_path
                start_epoch = load_from_epoch

            model_state = torch.load(model_path, map_location=torch.device(self.device))
            self.generator_nn.load_state_dict(model_state["generator"])
            # self.anomaly_detector_nn.apply(weights_init)
            self.anomaly_detector_nn.load_state_dict(model_state["discriminator"])
            self.disc_optimizer = torch.optim.Adam(
                self.anomaly_detector_nn.parameters(), lr=self.model.disc_lr
            )
            self.gen_optimizer = torch.optim.Adam(
                self.generator_nn.parameters(), lr=self.model.gen_lr
            )
            self.gen_optimizer.load_state_dict(model_state["gen_optimizer"])
            self.disc_optimizer.load_state_dict(model_state["disc_optimizer"])

        discriminator_scheduler = MultiStepLR(
            self.disc_optimizer, milestones=disc_milestones
        )
        generator_scheduler = MultiStepLR(self.gen_optimizer, milestones=gen_milestones)

        self.generator_nn.train()
        self.anomaly_detector_nn.eval()
        pin_memory = True if self.device == "cuda" else False
        init_train_loader = DataLoader(
            self.normal_train,
            batch_size=init_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory_device=self.device,
            pin_memory=pin_memory,
            num_workers=self.num_workers,
        )
        init_outliers_loader = DataLoader(
            self.outliers_dataset,
            batch_size=init_batch_size,
            sampler=RandomSampler(
                self.outliers_dataset, replacement=True, num_samples=int(1e10)
            ),
            pin_memory_device=self.device,
            pin_memory=pin_memory,
            num_workers=self.num_workers,
        )
        iterator = iter(init_outliers_loader)
        # -----------------------------TRAIN GENERATOR ALONE FOR N_INIT_EPOCHS-------------------------------
        for i in tqdm(range(start_epoch + 1, init_epochs + 1)):
            total_generator_loss = 0.0
            for normal_data in init_train_loader:
                # normal_data = next(iter(self.train_normal_loader))
                outliers = next(iterator)

                normal_data = normal_data.to(self.device)
                outliers = outliers.to(self.device)

                encoded_x = self.generator_nn.encoder_1(normal_data).to(self.device)
                encoded_a = self.generator_nn.encoder_1(outliers).to(self.device)
                images_vector = torch.cat((encoded_x, encoded_a), dim=1)
                max_values = images_vector.max(dim=1, keepdims=True).values
                min_values = images_vector.min(dim=1, keepdims=True).values
                normalized_vector = (images_vector - min_values) / (
                    max_values - min_values
                )

                # ------------------------------------------------------------------------------------------------------
                # ---------------------------------TRAIN GENERATOR----------------------------------------
                # -------------------------------------------------------------------------------------------------------

                gamma_1_input = torch.cat(
                    (
                        normalized_vector,
                        torch.ones(normal_data.size(0), 1, device=self.device),
                    ),
                    dim=1,
                ).view(-1, self.model.latent_dim_size, 1, 1)
                gamma_0_input = torch.cat(
                    (
                        normalized_vector,
                        torch.zeros(normal_data.size(0), 1, device=self.device),
                    ),
                    dim=1,
                ).view(-1, self.model.latent_dim_size, 1, 1)

                fake_outlier = (
                    self.generator_nn.generator_layers(gamma_1_input)
                    .squeeze(0)
                    .to(self.device)
                )
                fake_normal = (
                    self.generator_nn.generator_layers(gamma_0_input)
                    .squeeze(0)
                    .to(self.device)
                )
                gen_loss_outlier_value = gen_loss(fake_outlier, outliers)
                gen_loss_normal_value = gen_loss(fake_normal, normal_data)
                self.gen_optimizer.zero_grad()

                gen_loss_value = (
                    gen_loss_normal_value * self.model.lambda_1
                    + gen_loss_outlier_value * self.model.lambda_2
                )
                gen_loss_value.backward()
                self.gen_optimizer.step()
                total_generator_loss += gen_loss_value.item()

            avg_gen_loss = total_generator_loss / len(init_train_loader)
            if save_checkpoint is not None and i % save_checkpoint == 0:

                log_message = (
                    f"Epoch: {i} | G: loss {avg_gen_loss} | "
                    f"discriminator lr: {discriminator_scheduler.get_last_lr()} |  "
                    f"generator lr {generator_scheduler.get_last_lr()}\n"
                )
                self.__log(log_message)
                training_result = TrainingResult(
                    model=self.model.id, epoch=i, g_loss=avg_gen_loss, seed=self.seed
                )
                if self.save:
                    self.save_images(training_result)

            # avg_gen_loss = total_generator_loss / len(self.train_normal_loader)
            self.gen_train_losses.append(avg_gen_loss)

            generator_scheduler.step()

        starting_epoch = max(init_epochs, start_epoch)

        self.anomaly_detector_nn.train()

        for i in tqdm(range(starting_epoch + 1, epochs + 1)):
            total_discriminator_loss = 0.0
            total_generator_loss = 0.0
            total_grad_penalty = 0.0
            gen_train_counter = 0
            iterator = iter(self.outliers_loader)
            normal_iterator = iter(self.train_normal_loader)
            iterations_num = 100

            for batch_idx in range(1, iterations_num):
                normal_data = next(normal_iterator)
                # for batch_idx, normal_data in enumerate(self.train_normal_loader):
                #    normal_data = next(iter(self.train_normal_loader))
                outliers = next(iterator)

                normal_data = normal_data.to(self.device)
                outliers = outliers.to(self.device)

                encoded_x = self.generator_nn.encoder_1(normal_data).to(self.device)
                encoded_a = self.generator_nn.encoder_1(outliers).to(self.device)
                images_vector = torch.cat((encoded_x, encoded_a), dim=1)
                max_values = images_vector.max(dim=1, keepdims=True).values
                min_values = images_vector.min(dim=1, keepdims=True).values
                normalized_vector = (images_vector - min_values) / (
                    max_values - min_values
                )

                # -------------------------------------------------------------------------------------------------------
                # ---------------------------------------TRAIN DISCRIMINATOR -------------------------------------------
                # -------------------------------------------------------------------------------------------------------

                self.anomaly_detector_nn.zero_grad()
                interpolations = self.interpolation_distribution.sample(
                    (self.model.interpolations_sample_size, 1)
                ).to(self.device)
                img_mini_batch = normalized_vector.repeat(
                    self.model.interpolations_sample_size, 1
                )
                interpolations_batch = interpolations.repeat(
                    1, self.model.batch_size
                ).view(self.model.batch_size * self.model.interpolations_sample_size, 1)
                gen_input = (
                    torch.cat((img_mini_batch, interpolations_batch), dim=1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                generated_output = self.generator_nn.generator_layers(gen_input)
                self.disc_optimizer.zero_grad()
                disc_output = self.anomaly_detector_nn(generated_output)

                # ----------Clip Gradient of Anomaly Detector-------------

                # gradients = autograd.grad(
                #     outputs=disc_output,
                #     inputs=generated_output,
                #     grad_outputs=torch.ones(disc_output.size()).to(self.device),
                #     create_graph=True,
                #     retain_graph=True,
                # )[0]
                #
                # gradients = gradients.view(self.model.batch_size * self.model.interpolations_sample_size, -1)
                # grad_norm = gradients.norm(2, 1)
                # grad_penalty = torch.mean((grad_norm - 1) ** 2)
                grad_penalty = self.calculate_gradient_penalty(normalized_vector)
                anomaly_detector_estimated_loss = disc_loss(
                    disc_output, interpolations_batch
                )
                total_discriminator_loss += anomaly_detector_estimated_loss.item()
                anomaly_detector_estimated_loss += self.model.lambda_3 * grad_penalty
                total_grad_penalty += grad_penalty.item()

                # anomaly_detector_estimated_loss = disc_loss(disc_output, interpolations_batch)

                anomaly_detector_estimated_loss.backward(retain_graph=True)
                self.disc_optimizer.step()
                # total_discriminator_loss += anomaly_detector_estimated_loss.item()

                # ---------------------------------------------------------------------------------------------
                # ---------------------------------TRAIN GENERATOR EVERY N_CRITIC ITERATIONS-------------------
                # ---------------------------------------------------------------------------------------------
                if batch_idx % n_critic == 0:
                    self.generator_nn.zero_grad()
                    gamma_1_input = torch.cat(
                        (
                            normalized_vector,
                            torch.ones(normal_data.size(0), 1, device=self.device),
                        ),
                        dim=1,
                    ).view(-1, self.model.latent_dim_size, 1, 1)
                    gamma_0_input = torch.cat(
                        (
                            normalized_vector,
                            torch.zeros(normal_data.size(0), 1, device=self.device),
                        ),
                        dim=1,
                    ).view(-1, self.model.latent_dim_size, 1, 1)

                    fake_outlier = (
                        self.generator_nn.generator_layers(gamma_1_input)
                        .squeeze(0)
                        .to(self.device)
                    )
                    fake_normal = (
                        self.generator_nn.generator_layers(gamma_0_input)
                        .squeeze(0)
                        .to(self.device)
                    )
                    gen_loss_outlier_value = gen_loss(fake_outlier, outliers)
                    gen_loss_normal_value = gen_loss(fake_normal, normal_data)

                    gen_loss_value = (
                        gen_loss_normal_value * self.model.lambda_1
                        + gen_loss_outlier_value * self.model.lambda_2
                    )
                    self.gen_optimizer.zero_grad()
                    gen_loss_value.backward()
                    self.gen_optimizer.step()
                    total_generator_loss += gen_loss_value.item()
                    del fake_outlier
                    del fake_normal
                    gen_train_counter += 1

            avg_disc_loss = total_discriminator_loss / iterations_num
            avg_gen_loss = total_generator_loss / gen_train_counter
            avg_grad_penalty = total_grad_penalty / iterations_num
            self.gen_train_losses.append(avg_gen_loss)
            self.d_train_losses.append(avg_disc_loss)

            discriminator_scheduler.step()
            generator_scheduler.step()

            if save_checkpoint is not None and i % save_checkpoint == 0:
                log_message = (f"Epoch: {i} | D: loss {avg_disc_loss} | "
                               f"G: loss {avg_gen_loss} | discriminator lr: {discriminator_scheduler.get_last_lr()} | "
                               f"generator lr {generator_scheduler.get_last_lr()} | "
                               f"Avg gradient penalty: {avg_grad_penalty}\n")
                self.__log(log_message)
                training_result = TrainingResult(
                    model=self.model.id,
                    epoch=i,
                    g_loss=avg_gen_loss,
                    d_loss=avg_disc_loss,
                    seed=self.seed,
                )
                self.validate_model(training_result)
                f1_score = self.calculate_metrics()
                self.__log(f"F1 score: {f1_score}\n")
                if self.save:
                    self.save_images(training_result)
                    self.save_model(training_result)
                    with db:
                        training_result.save()

            if validation_checkpoint is not None and i % validation_checkpoint == 0:
                training_result = TrainingResult(
                    model=self.model,
                    epoch=i,
                    g_loss=avg_gen_loss,
                    d_loss=avg_disc_loss,
                    seed=self.seed,
                )
                self.validate_model(training_result)
                val_loss = training_result.d_val_loss
                if (
                    self.best_validation_result is None
                    or self.best_validation_result[0] < val_loss
                ):
                    self.best_validation_result = (
                        val_loss,
                        i,
                        self.generator_nn.state_dict(),
                        self.anomaly_detector_nn.state_dict(),
                    )

            torch.cuda.empty_cache()

        if self.save:
            self.save_model()

    def __log(self, log_message):
        if not self.save:
            return
        log_message += "\n"
        print(log_message)
        with open(self.__log_file_name, "a") as log_file:
            log_file.write(log_message)

    def save_model(self, training_result: TrainingResult = None):
        if training_result:
            filename = os.path.join(
                self.get_epoch_dir(training_result.epoch), "model.pt"
            )
            training_result.weights_path = filename

        else:
            Path(os.path.join(self.__output_path, str(self.seed))).mkdir(
                exist_ok=True, parents=True
            )
            filename = os.path.join(self.__output_path, str(self.seed), "model.pt")
            self.model.weights_path = filename
            with db:
                self.model.save()
        torch.save(
            {
                "generator": self.generator_nn.state_dict(),
                "discriminator": self.anomaly_detector_nn.state_dict(),
                "gen_optimizer": self.gen_optimizer.state_dict(),
                "disc_optimizer": self.disc_optimizer.state_dict(),
            },
            filename,
        )

    def get_epoch_dir(self, epoch):
        dir_path = os.path.join(self.__output_path, str(self.seed), str(epoch))
        Path(dir_path).mkdir(exist_ok=True, parents=True)
        return dir_path

    def save_images(self, training_result):
        with torch.no_grad():
            self.generator_nn.eval()
            self.anomaly_detector_nn.eval()
            step = 0.1
            interpolations_num = 11

            test_size = 5
            interpolations = torch.arange(
                0, interpolations_num * step, step=step, device=self.device
            ).view(interpolations_num, 1)
            interpolations_repeated = interpolations.repeat(test_size, 1).view(-1, 1)

            normal_data_loader = DataLoader(self.normal_train, batch_size=test_size)
            normal_data = next(iter(normal_data_loader))
            outliers_loader = DataLoader(
                self.outliers_dataset,
                sampler=RandomSampler(
                    self.outliers_dataset, replacement=True, num_samples=100
                ),
                batch_size=test_size,
            )
            outliers = next(iter(outliers_loader))
            _, axs = plt.subplots(test_size * 2, 13, figsize=(32, 32))
            axs = axs.flatten()

            normal_data = normal_data.to(self.device)
            outliers = outliers.to(self.device)

            encoded_1 = self.generator_nn.encoder_1(normal_data).to(self.device)
            encoded_2 = self.generator_nn.encoder_1(outliers).to(self.device)
            images_vector = torch.cat((encoded_1, encoded_2), dim=1)
            max_values = images_vector.max(dim=1, keepdims=True).values
            min_values = images_vector.min(dim=1, keepdims=True).values
            normalized_vector = (images_vector - min_values) / (max_values - min_values)
            normalized_vectors_batch = normalized_vector.repeat(
                1, interpolations_num
            ).view(test_size * interpolations_num, -1)
            gen_input = (
                torch.cat((normalized_vectors_batch, interpolations_repeated), dim=1)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            generated_images = self.generator_nn.generator_layers(gen_input)
            predictions = self.anomaly_detector_nn(generated_images)
            output = list(zip(generated_images, interpolations_repeated, predictions))

            images_to_display = []

            for idx, data in enumerate(normal_data):
                outlier = outliers[idx]
                images_to_display.append((data, "normal"))
                start_idx = idx * interpolations_num
                end_idx = (idx + 1) * interpolations_num
                generated = output[start_idx:end_idx]
                images_to_display.extend(
                    [
                        (
                            img,
                            f"Gamma: {gamma.squeeze(0).squeeze(0): .3f}, Pred: {float(prediction): .3f}",
                        )
                        for img, gamma, prediction in generated
                    ]
                )
                images_to_display.append((outlier, "outlier"))
                images_to_display.append((None, ""))
                sanity_checks = [
                    (
                        gamma * outlier + (1 - gamma) * data,
                        f"Sanity check: {gamma.squeeze(0).squeeze(0):.3f}",
                    )
                    for gamma in interpolations
                ]
                images_to_display.extend(sanity_checks)
                images_to_display.append((None, ""))

            del interpolations
            del interpolations_repeated

            for (img, title), ax in zip(images_to_display, axs):
                if img is not None:
                    orig_image = self.reverse_transform(img.view(3, 32, 32))
                    ax.imshow(orig_image)
                    ax.set_title(title, fontsize=6)
                else:
                    ax.axis("off")
            # plt.suptitle(f"D error: {training_result.d_loss:.3f}, G error {training_result.g_loss:.3f}")
            file_path = os.path.join(
                self.get_epoch_dir(training_result.epoch), "results.pdf"
            )
            plt.savefig(file_path)

            plt.close()
        self.generator_nn.train()
        self.anomaly_detector_nn.train()
        training_result.image_path = file_path

    def validate_model(self, training_result: TrainingResult):
        self.generator_nn.eval()
        self.anomaly_detector_nn.eval()
        with torch.no_grad():
            total_gen_loss = 0.0
            total_disc_loss = 0.0
            for normal_data in self.val_normal_loader:
                # for _ in range(2):
                #    normal_data = next(iter(self.val_normal_loader))
                outliers = next(iter(self.outliers_loader))

                normal_data = normal_data.to(self.device)
                outliers = outliers.to(self.device)

                interpolations = self.interpolation_distribution.sample(
                    (self.model.interpolations_sample_size, 1)
                ).to(self.device)
                encoded_x = self.generator_nn.encoder_1(normal_data).to(self.device)
                encoded_a = self.generator_nn.encoder_1(outliers).to(self.device)
                images_vector = torch.cat((encoded_x, encoded_a), dim=1)
                max_values = images_vector.max(dim=1, keepdims=True).values
                min_values = images_vector.min(dim=1, keepdims=True).values
                normalized_vector = (images_vector - min_values) / (
                    max_values - min_values
                )

                img_mini_batch = images_vector.repeat(
                    self.model.interpolations_sample_size, 1
                )
                interpolations_batch = interpolations.repeat(
                    1, self.model.batch_size
                ).view(-1, 1)
                gen_input = (
                    torch.cat((img_mini_batch, interpolations_batch), dim=1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                generated_output = self.generator_nn.generator_layers(gen_input)
                disc_output = self.anomaly_detector_nn(generated_output)
                anomaly_detector_estimated_loss = disc_loss(
                    disc_output, interpolations_batch
                )

                gamma_1_input = torch.cat(
                    (
                        normalized_vector,
                        torch.ones(normal_data.size(0), 1, device=self.device),
                    ),
                    dim=1,
                ).view(-1, self.model.latent_dim_size, 1, 1)
                gamma_0_input = torch.cat(
                    (
                        normalized_vector,
                        torch.zeros(normal_data.size(0), 1, device=self.device),
                    ),
                    dim=1,
                ).view(-1, self.model.latent_dim_size, 1, 1)

                fake_outlier = (
                    self.generator_nn.generator_layers(gamma_1_input)
                    .squeeze(0)
                    .to(self.device)
                )
                fake_normal = (
                    self.generator_nn.generator_layers(gamma_0_input)
                    .squeeze(0)
                    .to(self.device)
                )
                gen_loss_outlier_value = gen_loss(fake_outlier, outliers)
                gen_loss_normal_value = gen_loss(fake_normal, normal_data)

                gen_loss_value = (
                    gen_loss_normal_value * self.model.lambda_1
                    + gen_loss_outlier_value * self.model.lambda_2
                )

                total_gen_loss += gen_loss_value.item()
                del fake_outlier
                del fake_normal
                total_disc_loss += anomaly_detector_estimated_loss.item()
            total_gen_loss /= len(self.val_normal_loader)
            total_disc_loss /= len(self.val_normal_loader)
            training_result.g_val_loss = total_gen_loss
            training_result.d_val_loss = total_disc_loss
            self.__log(
                f"Generator validation loss: {total_gen_loss}, Discriminator validation loss: {total_disc_loss}"
            )
        self.generator_nn.train()
        self.anomaly_detector_nn.train()

    def calculate_metrics(self):
        self.anomaly_detector_nn.eval()
        predictions = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([])
        with torch.no_grad():
            for data, labels in self.test_loader:
                data = data.to(self.device)
                prediction = self.anomaly_detector_nn(data)
                predictions = torch.cat((predictions, prediction))
                ground_truth = torch.cat((ground_truth, labels))

        predictions = predictions.cpu().view_as(ground_truth)
        precision, recall, thresholds = precision_recall_curve(
            ground_truth, predictions
        )

        f1_scores = calculate_f1_scores(predictions, ground_truth, thresholds)
        optimal_f1_score = max(f1_scores)
        self.anomaly_detector_nn.train()
        return optimal_f1_score

    def calculate_gradient_penalty(self, normalized_vector):
        alpha = self.interpolation_distribution.sample(
            (normalized_vector.size(0), 1)
        ).to(self.device)
        gen_input = (
            torch.cat((normalized_vector, alpha), dim=1).unsqueeze(-1).unsqueeze(-1)
        )
        interpolations = self.generator_nn.generator_layers(gen_input)
        interpolations.requires_grad_(True)
        disc_output = self.anomaly_detector_nn(interpolations)
        gradients = torch.autograd.grad(
            outputs=disc_output,
            inputs=interpolations,
            grad_outputs=torch.ones(disc_output.size(), device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        return gradient_penalty

    def disc_loss(self, disc_pred, ground_truth_val):
        criterion = nn.MSELoss(reduction="none")
        ground_truth = ground_truth_val.view(disc_pred.shape)
        loss = criterion(disc_pred, ground_truth)
        log_prob = self.interpolation_distribution.log_prob(ground_truth_val)
        probs = torch.exp(log_prob)
        loss_val = torch.dot(loss.view(-1), probs.view(-1))
        return loss_val

    def __del__(self):
        # cleanup()
        pass
