from torch import nn
import torch
from torch.nn import functional as F


from torch import nn


class CIFARDiscriminator(nn.Module):

    def __init__(self):
        super(CIFARDiscriminator, self).__init__()

        self.block_1 = get_initial_disc_block(3, 32, kernel_size=3, stride=1, padding=1)
        self.block_2 = get_disc_block(32, 64, kernel_size=3, stride=2, padding=1)
        self.block_3 = get_disc_block(64, 128, kernel_size=3, stride=1, padding=1)
        self.block_4 = get_disc_block(128, 256, kernel_size=3, stride=2, padding=1)
        self.block_5 = get_disc_block(256, 512, kernel_size=4, stride=2, padding=1)
        self.block_6 = get_final_disc_block(512, 1, kernel_size=4, stride=1)


    def forward(self, point):
        x_1 = self.block_1(point)
        x_2 = self.block_2(x_1)
        x_3 = self.block_3(x_2)
        x_4 = self.block_4(x_3)
        x_5 = self.block_5(x_4)
        x_6 = self.block_6(x_5)

        return x_6

def get_disc_block(in_channels, out_channels, kernel_size, stride, padding = 0):
    block = [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
        nn.Dropout(0.1),
        #nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),

    ]
    return nn.Sequential(*block)


def get_initial_disc_block(in_channels, out_channels, kernel_size, stride, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2)
    )


def get_final_disc_block(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
        #nn.Dropout(0.3),
        nn.Flatten(),
        nn.Sigmoid()
    )


discriminator_nn = CIFARDiscriminator()