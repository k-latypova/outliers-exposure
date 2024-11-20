from typing import List

import click
import torch.cuda
from sklearn.model_selection import GridSearchCV

from src.utils.cifar_models import CIFARModel, db
from trainer import MixupTrainer
from bayes_opt import BayesianOptimization
from src.models.cifar_model import CIFARTrainingModel
from src.models.cifar_evaluator import CIFARScorer
import pandas as pd


def objective_function(lambda_one: float, lambda_two: float, gen_lr: float, disc_lr: float):
    num_epochs = 300
    normal_dataset_size = 300
    outliers_dataset_size = 50
    interpolation_sample_size = 1000
    trainer = MixupTrainer(num_epochs, 32, outliers_dataset_size=outliers_dataset_size,
                           normal_dataset_size=normal_dataset_size,
                           interpolation_sample_size=interpolation_sample_size,
                           lambda_1=lambda_one,
                           lambda_2=lambda_two,
                           device="cuda",
                           generator_lr=gen_lr,
                           discriminator_lr=disc_lr)

    trainer.train()
    auc = trainer.score()
    return auc


def scoring_function(function_outputs):
    return -function_outputs


def tune_hyperparameters_bo():
    pbounds = {'lambda_one': (5e-4, 9e-3), 'lambda_two': (5e-4, 9e-3),
               'gen_lr': (1e-5, 1e-3), 'disc_lr': (1e-5, 1e-3), 'n_critic': (5, 10)}
    optimizer = BayesianOptimization(objective_function, pbounds=pbounds, random_state=42)
    optimizer.probe(
        params={'lambda_one': 0.005, 'lambda_two': 0.005,
                'gen_lr': 0.0001, 'disc_lr': 0.0001},
        lazy=True,
    )
    optimizer.maximize(init_points=3, n_iter=8)
    best_params = optimizer.max['params']
    best_auc = optimizer.max['target']
    print("Best hyperparameters with BO:", best_params)
    print("Best AUC value with BO:", best_auc)


def tune_hyperparameters_grid_search():
    # Define the parameter grid
    param_grid = {'lambda_one': [0.001, 0.003, 0.005, 0.007, 0.009], 'lambda_two': [0.001, 0.003, 0.005, 0.007, 0.009],
                'gen_lr': [0.001, 0.002, 0.0001, 0.0002], 'disc_lr': [0.001, 0.002, 0.0001, 0.00001]}
    cv = GridSearchCV(estimator=objective_function, param_grid=param_grid, scoring=scoring_function)
    cv.fit()
    best_params = cv.best_params_
    best_auc = cv.best_score_
    print("Best parameters with grid search:", best_params)
    print("Best AUC with:", best_auc)


@click.command()
@click.option("--device", type=str, default="cpu")
def bayesian_optimization_cifar_model(device):
    def objective_function(lambda_one: float, lambda_two: float, gen_lr: float, disc_lr: float, n_critic_val):
        n_critic = int(n_critic_val)
        model = CIFARModel(description="", outliers_num=25,
                                            normal_data_num=500,
                                            lambda_1=lambda_one, lambda_2=lambda_two,
                                            gen_lr=gen_lr, disc_lr=disc_lr,
                                            gen_lr_milestones='', disc_lr_milestones='',
                                            interpolations_sample_size=1000,
                                            outliers_label=3, normal_label=1, batch_size=32)
        training_model = CIFARTrainingModel(model, device, save=False)
        training_model.train(1000, 4, 42, n_critic=n_critic, validation_checkpoint=25)
        print(f"Evaluating on a epoch: {training_model.best_validation_result[1]} with best validation loss of {training_model.best_validation_result[0]}", flush=True)
        score = CIFARScorer(training_model.best_validation_result[2], training_model.best_validation_result[3], model, 42, device).score(4)
        return score


    pbounds = {'lambda_one': (5e-4, 9e-2), 'lambda_two': (5e-4, 9e-2),
               'gen_lr': (1e-5, 1e-3), 'disc_lr': (1e-5, 1e-3), 'n_critic_val': (5, 10)}

    optimizer = BayesianOptimization(objective_function, pbounds=pbounds, random_state=42)
    optimizer.probe(
        params={'lambda_one': 0.005, 'lambda_two': 0.005,
                'gen_lr': 0.0001, 'disc_lr': 0.0001, 'n_critic_val': 5},
        lazy=True,
    )
    optimizer.maximize(init_points=3, n_iter=8)
    best_params = optimizer.max['params']
    best_auc = optimizer.max['target']
    print("Best hyperparameters with BO:", best_params)
    print("Best AUC value with BO:", best_auc)



# @click.command()
# @click.option("--d_lr", type=float, help="discriminator lr")
# @click.option("--g_lr", type=float, help="generator lr")
# @click.option("--lambda_one", type=float, help="lambda one")
# @click.option("--lambda_two", type=float, help="lambda two")
# @click.option("--n_critic", type=int, help="n critic")
def objective_function_for_grid_search(args):
    g_lr, d_lr, lambda_one, lambda_two, n_critic, device, num_workers, vector_dim, normal_label, anomaly_label, test_label = args
    print(f"Starting to train a model with lambda_1: {lambda_one}, lambda_2: {lambda_two}, g_lr: {g_lr}, d_lr: {d_lr}, n_critic: {n_critic}, device: {device}\n", flush=True)
    print(f" Cuda is available: {torch.cuda.is_available()}\n", flush=True)
    sample_size = 16384
    #sample_size= 16
    interpolations_num = 512
    batch_size = sample_size / interpolations_num
    model = CIFARModel(description="", outliers_num=100,
                                            normal_data_num=500,
                       normal_label=-1, outliers_label=-1,
                       latent_dim_size=int(vector_dim * 2 + 1),
                                            lambda_1=lambda_one, lambda_2=lambda_two,
                                            gen_lr=g_lr, disc_lr=d_lr,
                                            gen_lr_milestones='150,300,450,600', disc_lr_milestones='450,600,800',
                                            interpolations_sample_size=interpolations_num,
                                            outliers_labels=str(anomaly_label), normal_labels=str(normal_label), batch_size=int(batch_size))
    training_model = CIFARTrainingModel(model, device, save=False, num_workers=num_workers)
    with db:
        model.save()
    num_epochs = 350
    seed = 42
    init_epochs = 150
    training_model.train(num_epochs, seed, init_epochs, n_critic=n_critic, validation_checkpoint=10, num_workers=num_workers, init_batch_size=256, encoder_name=f"model_cifar_{int(vector_dim)}_best.pt",
                         normal_data_num=1000)
    print(f"Evaluating on a epoch: {training_model.best_validation_result[1]} with best validation loss of {training_model.best_validation_result[0]}\n", flush=True)
    score = CIFARScorer(training_model.best_validation_result[2], training_model.best_validation_result[3], model, 42, device, training_model.transform, num_workers).score(test_label)
    #score = CIFARScore(model_id=model.id, score=score, epoch=training_model.best_validation_result[1], seed=42)
    # with db:
    #     score.save()
    #training_model.save_model()
    
    return score


@click.command()
# @click.option("--d_lr", "-dlr", type=float, help="discriminator lr", multiple=True)
# @click.option("--g_lr", "-glr", type=float, help="generator lr", multiple=True)
# @click.option("--lambda_one", "-l1", type=float, help="lambda one", multiple=True)
# @click.option("--lambda_two", "-l2", type=float, help="lambda two", multiple=True)
# @click.option("--n_critic", "-nc", type=int, help="n critic", multiple=True)
# @click.option("--interpolations_num", type=int, default=1000)
@click.option("--normal_label", type=int)
@click.option("--anomaly_label", type=int)
@click.option("--test_label", type=int)
@click.option("--device", type=str, default="cuda")
@click.option("--num_workers", type=int, default=1)
def grid_search_cifar(normal_label, anomaly_label, test_label, device: str, num_workers):
    def apply_row(row):
        score = objective_function_for_grid_search((row["g_lr"], row["d_lr"], row["lambda_1"], row["lambda_2"],
                                                    row["n_critic"], device, num_workers, row["vector_dim"],
                                                    normal_label, anomaly_label, test_label))
        return score
    filename = "params.csv"
    df = pd.read_csv(filename, sep=',')

    df["f1"] = df.apply(lambda row: apply_row(row), axis=1)
    df.to_csv(f"scores_{normal_label}_{anomaly_label}.csv", index=False)

if __name__ == "__main__":
    grid_search_cifar()
