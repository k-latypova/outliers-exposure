from datetime import datetime

from src.models.cifar_model import CIFARTrainingModel
import click
from src.utils.cifar_models import CIFARModel, init_db


@click.command()
@click.option(
    "--g_lr",
    type=float,
    default=0.001,
    help="Initial learning rate for the network training. Default=0.001",
)
@click.option(
    "--d_lr",
    type=float,
    default=0.001,
    help="Initial learning rate for the network training. Default=0.001",
)
@click.option("--n_epochs", type=int, default=100, help="Number of epochs to train.")
@click.option(
    "--batch_size", type=int, default=128, help="Batch size for mini-batch training."
)
@click.option("--lambda_one", type=float, default=0.5, help="Lambda 1 in loss function")
@click.option("--lambda_two", type=float, default=0.5, help="Lambda 2 in loss function")
@click.option(
    "--lambda_three", type=float, default=0.5, help="Gradient penalty coefficient"
)
@click.option("--outliers_num", type=int, default=1, help="how many outliers")
@click.option(
    "--normal_dataset_size",
    type=int,
    default=128,
    help="how many samples in normal dataset",
)
@click.option(
    "--interpolations_num", type=int, default=100, help="how many interpolations"
)
@click.option(
    "--device", type=str, default="cpu", help="device on which to run tensors"
)
# @click.option('--normal_label', type=int, default=1, help='normal data label')
# @click.option('--outlier_label', type=int, default=2, help='outlier label')
@click.option("--normal_labels", type=str, help="normal data labels")
@click.option("--outliers_labels", type=str, help="outliers labesl")
# @click.option('--test_label', type=int, default=3, help='test outlier label')
@click.option(
    "--g_milestones", type=str, default="", help="generator scheduler lr milestones"
)
@click.option(
    "--d_milestones", type=str, default="", help="discriminator scheduler lr milestones"
)
@click.option(
    "--description", type=str, default="", help="description of model and training"
)
@click.option(
    "--save_checkpoint", type=int, default=10, help="save model every n epochs"
)
@click.option(
    "--n_critic",
    type=int,
    default=5,
    help="how many times update the discriminator weights before training generator",
)
@click.option("--cuda_no", type=str)
@click.option("--num_workers", type=int, default=1, help="Number of workers")
@click.option("--init_epochs", type=int, default=300, help="Init epochs")
@click.option("--seed", type=int, default=42, help="Seed")
@click.option(
    "--init_batch_size",
    type=int,
    default=16,
    help="Batch size for initial generator training",
)
@click.option(
    "--encoder_name",
    type=str,
    help="Name of the encoder to convert images to latent space",
)
@click.option(
    "--vector_dim", type=int, default=256, help="Dimension of the latent vector"
)
def main(
    g_lr,
    d_lr,
    n_epochs,
    batch_size,
    lambda_one,
    lambda_two,
    outliers_num,
    normal_dataset_size,
    interpolations_num,
    device,
    g_milestones,
    d_milestones,
    description,
    normal_labels,
    outliers_labels,
    save_checkpoint,
    n_critic,
    init_epochs,
    seed,
    init_batch_size,
    vector_dim,
    lambda_three,
    cuda_no=None,
    num_workers=None,
    encoder_name=None,
):
    init_db()
    latent_dim_size = vector_dim * 2 + 1
    model = CIFARModel(
        description=description,
        outliers_num=outliers_num,
        normal_data_num=normal_dataset_size,
        lambda_1=lambda_one,
        lambda_2=lambda_two,
        gen_lr=g_lr,
        disc_lr=d_lr,
        gen_lr_milestones=g_milestones,
        disc_lr_milestones=d_milestones,
        created_at=datetime.utcnow(),
        interpolations_sample_size=interpolations_num,
        normal_labels=normal_labels,
        outliers_labels=outliers_labels,
        batch_size=batch_size,
        normal_label=-1,
        outliers_label=-1,
        latent_dim_size=latent_dim_size,
        lambda_3=lambda_three,
    )
    training_model = CIFARTrainingModel(model, device, num_workers=num_workers)
    training_model.train(
        n_epochs,
        seed,
        init_epochs,
        save_checkpoint=save_checkpoint,
        n_critic=n_critic,
        num_workers=None,
        encoder_name=encoder_name,
        init_batch_size=init_batch_size,
    )


if __name__ == "__main__":
    main()
