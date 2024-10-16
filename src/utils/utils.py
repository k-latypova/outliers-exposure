import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


def init_weights_xavier(module):
    if isinstance(module, nn.Conv2d):
        torch.nn.init.xavier_uniform_(module.weight)
        # module.bias.data.fill_(0.01)
    elif isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
    # module.bias.data.fill_(0.01)


def disc_loss(disc_pred, ground_truth_val):
    criterion = nn.MSELoss()
    ground_truth = ground_truth_val.view(disc_pred.shape)
    loss = criterion(disc_pred, ground_truth)
    return loss


def gen_loss_1(gen_point, ground_truth):
    criterion = nn.MSELoss()
    ground_truth = ground_truth.view(gen_point.shape)
    loss = criterion(gen_point.float(), ground_truth.float())
    return loss


def gen_loss(output, batch):
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def apply_threshold(probabilities, threshold):
    mask = probabilities >= threshold
    labels = mask.int()
    return labels


def calculate_f1_scores(predictions, ground_truth, thresholds):
    f1_scores = []
    for threshold in thresholds:
        thresholded_predictions = apply_threshold(predictions, threshold)
        f1_score_val = f1_score(ground_truth, thresholded_predictions)
        f1_scores.append(f1_score_val)

    return f1_scores
