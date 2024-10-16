import torch
from torch import nn


class CifarGenerator(nn.Module):
    def __init__(self, encoded_img_length, latent_noise_dim):
        super(CifarGenerator, self).__init__()
        self.encoded_img_dim = encoded_img_length
        self.latent_dim = latent_noise_dim

        self.generator_layers = nn.Sequential(
            get_gen_block(self.latent_dim, 256, kernel_size=4, stride=1),
            get_gen_block(256, 128, kernel_size=4, stride=2, padding=1),
            get_gen_block(128, 64, kernel_size=4, stride=2, padding=1),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        c_hid = 32
        num_input_channels = 3
        act_fn = nn.ReLU

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            #nn.BatchNorm2d(c_hid),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            #nn.BatchNorm2d(2 * c_hid),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, encoded_img_length)
        )

    def forward(self, normal_image, outlier, gammas):
        normal_image_vector = self.encoder_1(normal_image.view(-1, 3, 32, 32))
        outlier_vector = self.encoder_1(outlier.view(-1, 3, 32, 32))

        images_vector = torch.cat((normal_image_vector, outlier_vector), dim=1)
        max_values = images_vector.max(dim=1, keepdims=True).values
        min_values = images_vector.min(dim=1, keepdims=True).values
        normalized_vector = (images_vector - min_values) / (max_values - min_values)
        gammas = gammas.view(-1, normal_image_vector.size(0), 1)
        outputs = torch.empty(size=(gammas.size(0), normal_image_vector.size(0), 3, 32, 32))
        # gamma = gamma.view(-1, 1)
        # generation_input = torch.cat((normalized_vector, gamma), dim=1).view(-1, self.latent_dim, 1, 1)
        # return self.generator_layers(generation_input)

        for idx, gamma in enumerate(gammas):
            gamma = gamma.view(-1, 1)
            generation_input = torch.cat((normalized_vector, gamma), dim=1).view(-1, self.latent_dim, 1, 1)
            outputs[idx] = self.generator_layers(generation_input)
        return outputs



def get_gen_block(in_channels, out_channels, kernel_size, stride, output_activation=False, padding=0):
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding),
        nn.ReLU()
    ]
    if output_activation:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


generator_nn = CifarGenerator(256, 513)




class CifarGenerator(nn.Module):
    def __init__(self, encoded_img_length, latent_noise_dim):
        super(CifarGenerator, self).__init__()
        self.encoded_img_dim = encoded_img_length
        self.latent_dim = latent_noise_dim

        def get_gen_block(in_channels, out_channels, kernel_size, stride, output_activation=False, padding=0):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding),
                nn.ReLU()
            ]
            if output_activation:
                layers.append(nn.Tanh())
            return nn.Sequential(*layers)

        self.generator_layers = nn.Sequential(
            get_gen_block(self.latent_dim, 256, kernel_size=4, stride=1),
            get_gen_block(256, 128, kernel_size=4, stride=2, padding=1),
            get_gen_block(128, 64, kernel_size=4, stride=2, padding=1),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        c_hid = 32
        num_input_channels = 3
        act_fn = nn.ReLU

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            #nn.BatchNorm2d(c_hid),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            #nn.BatchNorm2d(2 * c_hid),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, encoded_img_length)
        )

    def forward(self, normal_image, outlier, gammas):
        normal_image_vector = self.encoder_1(normal_image.view(-1, 3, 32, 32))
        outlier_vector = self.encoder_1(outlier.view(-1, 3, 32, 32))

        images_vector = torch.cat((normal_image_vector, outlier_vector), dim=1)
        max_values = images_vector.max(dim=1, keepdims=True).values
        min_values = images_vector.min(dim=1, keepdims=True).values
        normalized_vector = (images_vector - min_values) / (max_values - min_values)
        gammas = gammas.view(-1, normal_image_vector.size(0), 1)
        outputs = torch.empty(size=(gammas.size(0), normal_image_vector.size(0), 3, 32, 32))
        # gamma = gamma.view(-1, 1)
        # generation_input = torch.cat((normalized_vector, gamma), dim=1).view(-1, self.latent_dim, 1, 1)
        # return self.generator_layers(generation_input)

        for idx, gamma in enumerate(gammas):
            gamma = gamma.view(-1, 1)
            generation_input = torch.cat((normalized_vector, gamma), dim=1).view(-1, self.latent_dim, 1, 1)
            outputs[idx] = self.generator_layers(generation_input)
        return outputs



def get_gen_block(in_channels, out_channels, kernel_size, stride, output_activation=False, padding=0):
    layers = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding),
        nn.ReLU()
    ]
    if output_activation:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


generator_nn = CifarGenerator(256, 513)

