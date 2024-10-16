import torch
from torch import nn


def get_gen_block(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU()
                         )


def get_final_gen_block(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
                         nn.Tanh())


class MNISTGenerator(nn.Module):

    def __init__(self, generator_vector_dim) -> None:
        super(MNISTGenerator, self).__init__()
        self.generator_vector_dim = generator_vector_dim

        self.conv_layers = nn.Sequential(nn.Conv2d(2, 8, 5, padding=2),
                                         nn.BatchNorm2d(8),
                                         nn.LeakyReLU(0.2),
                                         nn.MaxPool2d(2, 2),
                                         nn.Conv2d(8, 4, 5, padding=2),
                                         nn.BatchNorm2d(4),
                                         nn.LeakyReLU(0.2),
                                         nn.MaxPool2d(2, 2),
                                         nn.Flatten(),
                                         nn.Linear(4 * 7 * 7, self.generator_vector_dim - 1)

                                         )
        #self.linear =
        self.generator_layers = nn.Sequential(get_gen_block(self.generator_vector_dim, 256, 3, 2),
                                              get_gen_block(256, 128, 4, 1),
                                              get_gen_block(128, 64, 3, 2),
                                              get_gen_block(64, 1, 4, 2))

    def forward(self, input_vector, gamma):
        x = input_vector.view(-1, 2, 28, 28)
        x1 = self.conv_layers(x)
        #x2 = self.linear(x1)
        x2 = x1.view(x.size(0), 1, -1)
        gamma = gamma.view(-1, 1, 1)
        x3 = torch.cat((x2, gamma), dim=2)
        x3 = x3.view(-1, 33, 1, 1)
        x4 = self.generator_layers(x3)
        return x4



generator_nn = MNISTGenerator(33)
