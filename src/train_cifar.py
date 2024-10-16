from src.utils.cifar_models import CIFARModel, db
from models.cifar_model import CIFARTrainingModel
import click


@click.command()
@click.option("--id", type=int, help="ID of the model")
@click.option("--epoch", type=int, help="Num of last epoch")
@click.option("--test_outlier_label", type=int, default=4)
@click.option("--seed", type=int, default=42)
@click.option("--num_epochs", type=int, help="How many epochs to train")
@click.option(
    "--save_checkpoint",
    type=int,
    default=50,
    help="Every how many epochs save the model",
)
@click.option("--device", type=str, default="cpu", help="Device")
@click.option(
    "--n_critic", type=int, default=5, help="discriminator training frequency"
)
@click.option("--n_workers", type=int, default=None, help="number of workers")
@click.option("--init_epochs", type=int, default=300, help="Init epochs")
def train(
    id,
    epoch,
    test_outlier_label,
    seed,
    num_epochs,
    save_checkpoint,
    n_critic,
    device,
    init_epochs,
    n_workers=None,
):
    with db:
        model = CIFARModel.get(CIFARModel.id == id)
    training_model = CIFARTrainingModel(model, device)
    training_model.train(
        num_epochs,
        seed,
        save_checkpoint=save_checkpoint,
        n_critic=n_critic,
        load_from_model=True,
        load_from_epoch=epoch,
        num_workers=n_workers,
        init_epochs=init_epochs,
    )


if __name__ == "__main__":
    train()
