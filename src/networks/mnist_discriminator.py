from torch import nn


def get_disc_block(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2)
                         )


def get_final_disc_block(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                         nn.Sigmoid())


class MNISTDiscriminator(nn.Module):

    def __init__(self) -> None:
        super(MNISTDiscriminator, self).__init__()

        self.block_1 = get_disc_block(1, 16, (3, 3), 2)
        self.block_2 = get_disc_block(16, 32, (5, 5), 2)
        self.block_3 = get_final_disc_block(32, 1, (5, 5), 2)
        # self.block_4 = get_final_disc_block(64, 1, (3, 3), 2)
        # self.block_4 = nn.Flatten()
        # self.block_5 = nn.Linear(64, 1)
        # self.sigmoid = nn.Sigmoid()
        self.float()


    def forward(self, point):
        x_1 = self.block_1(point)
        x_2 = self.block_2(x_1)
        x_3 = self.block_3(x_2)
        # x_4 = self.block_4(x_3)
        # x_5 = self.block_5(x_4)
        # x_6 = self.sigmoid(x_5)
        #
        # return x_6
        return x_3




discriminator_nn = MNISTDiscriminator()
