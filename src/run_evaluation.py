import click

from models.cifar_new_evaluator import CifarEvaluator
from utils.cifar_models import CIFARModel, TrainingResult, db, init_db


def get_model_epoch(model_epoch_str):
    model_id, epoch = model_epoch_str.split(":")
    return int(model_id), int(epoch)


@click.command()
@click.option("--models", type=str, help="Model ID and training results")
@click.option("--test_label", type=int, help="Test labels")
@click.option("--num_workers", type=int, help="Number of workers", default=1)
@click.option("--device", type=str, default="cpu")
@click.option("--batch_size", type=int, default=128)
def evaluate(models, test_label, num_workers, device, batch_size):
    init_db()
    models_and_epochs = [get_model_epoch(x) for x in models.split(",")]
    with db:
        models = []
        for model_id, epoch in models_and_epochs:

            training_result = (
                TrainingResult.select()
                .join(CIFARModel)
                .where(CIFARModel.id == model_id, TrainingResult.epoch == epoch)
                .get()
            )
            model = training_result.model
            if training_result is None:
                raise Exception(
                    f"Training result for model {model_id} with epoch {epoch}"
                )
            models.append((model, training_result))

    evaluator = CifarEvaluator(models=models, device=device)
    evaluator.run(test_label, batch_size, num_workers=num_workers)


if __name__ == "__main__":
    evaluate()
