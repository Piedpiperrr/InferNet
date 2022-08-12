# Using GAN to generate fake MNIST images
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


img_size = 28
latent_dim = 100
channels = 1

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(
                latent_dim,
                128 * self.init_size ** 2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(channels, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

generator = Generator()
generator.load_state_dict(torch.load(r'./generator_params/epoch173.pth'))


for i in range(1):
    z = Variable(torch.randn(10000, 100))  # cuda
    gen_imgs = generator(z)
    gen_imgs = gen_imgs.detach().numpy()
    print(gen_imgs.shape)
    np.save('./dataset/generated_imgs_{}'.format(i+1), gen_imgs)
