from models.cifar_evaluator import CIFAREvaluator
from utils.cifar_models import CIFARModel, db
import click


@click.command()
@click.option("--model_id", type=int)
@click.option("--epoch", type=int)
@click.option("--device", type=str, default='cpu')
@click.option("--seed", type=int, default=42)
@click.option("--test_label", type=int)
def evaluate(model_id, device, seed, test_label, epoch=None):
    with db:
        model = CIFARModel.get(CIFARModel.id == model_id)
    evaluator = CIFAREvaluator(model, device, epoch=epoch, seed=seed)
    evaluator.evaluate(test_label)


if __name__ == "__main__":
    evaluate()
