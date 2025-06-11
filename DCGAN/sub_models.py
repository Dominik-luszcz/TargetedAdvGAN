import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, output_dim):
        super().__init__()

        self.epsilon = 0.2

        self.output_dim = output_dim


        self.generator = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, out_channels=128, kernel_size=5, stride=4), # 4 for full recording (2, 2, 2) for random 500 sample
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.25),

            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=5, stride=3), # 3
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.25),

            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=5, stride=4), # 4
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.25),

            nn.Conv1d(in_channels=128, out_channels=self.output_dim, kernel_size=5, stride=1, padding=5//2),
            nn.Tanh() # force returns between -1 and 1, then after multiply by epsilon to ensure its reasonable (between -0.2 and 0.2)
        )


        self.double()

    def forward(self, noise):
        noise = torch.permute(noise, (0, 2, 1))
        output = self.generator(noise)
        #output = output[:, :, :500] # will not hardcode it, fix this after
        output = output[:, :, :2433]
        output = output - torch.mean(output)
        output = 0.2 * output # enforce returns to be between -0.2 and 0.2
        return torch.permute(output, (0,2,1))

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.discriminator = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, stride=1, padding=5//2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=5, stride=1, padding=5//2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv1d(in_channels=hidden_dim * 2, out_channels=hidden_dim, kernel_size=5, stride=1, padding=5//2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv1d(in_channels=hidden_dim, out_channels=output_dim, kernel_size=5, stride=1, padding=5//2),
            nn.Sigmoid()
        )

        self.double()

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        output = self.discriminator(x)
        return torch.permute(output, (0, 2, 1))