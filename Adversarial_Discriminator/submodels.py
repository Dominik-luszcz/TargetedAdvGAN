import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.discriminator = nn.Sequential(
            nn.BatchNorm1d(num_features=input_dim),
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=3 // 2,
                stride=1,
            ),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim * 2,
                kernel_size=5,
                padding=5 // 2,
                stride=1,
            ),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(num_features=hidden_dim * 2),
            nn.Conv1d(
                in_channels=hidden_dim * 2,
                out_channels=hidden_dim,
                kernel_size=5,
                padding=5 // 2,
                stride=1,
            ),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(num_features=hidden_dim),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=output_dim,
                kernel_size=3,
                padding=3 // 2,
                stride=1,
            ),
        )

        self.double()

    def forward(self, data):

        x = data

        x = torch.permute(x, (0, 2, 1))
        x = self.discriminator(x)
        x = torch.permute(x, (0, 2, 1))
        return x.mean(dim=1)


class TCN_Block(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation, padding):
        super().__init__()
        self.padding = padding

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=input_dim,
                out_channels=output_dim,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
            )
        )
        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels=output_dim,
                out_channels=output_dim,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
            )
        )
        self.dropout2 = nn.Dropout(p=0.2)

        self.downsample = nn.Conv1d(
            in_channels=input_dim, out_channels=output_dim, kernel_size=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # Do the first dialated conv
        tcn_output = self.conv1(x)
        # remove excess padding
        tcn_output = tcn_output[:, :, : -self.padding]
        tcn_output = self.leaky(tcn_output)
        tcn_output = self.dropout1(tcn_output)

        # do the second dilated conv
        tcn_output = self.conv2(tcn_output)
        tcn_output = tcn_output[:, :, : -self.padding]
        tcn_output = self.leaky(tcn_output)
        tcn_output = self.dropout2(tcn_output)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(tcn_output + x)
