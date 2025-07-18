import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, output_dim, sample_size):
        super().__init__()

        self.epsilon = 0.2

        self.output_dim = output_dim
        self.sample_size = sample_size

        self.generator = nn.Sequential(
            TCN_Block(input_dim=1, output_dim=64, kernel_size=3, dilation=1, padding=(3-1) * 1),
            TCN_Block(input_dim=64, output_dim=128, kernel_size=5, dilation=2, padding=(5-1) * 2),
            TCN_Block(input_dim=128, output_dim=64, kernel_size=5, dilation=4, padding=(5-1) * 4),
            TCN_Block(input_dim=64, output_dim=32, kernel_size=3, dilation=8, padding=(3-1) * 8),
        )
        self.output = nn.Linear(32, 1)



        self.double()

    def forward(self, noise):
        noise = torch.permute(noise, (0, 2, 1))
        output = self.generator(noise)
        output = torch.permute(output, (0, 2, 1))
        output = self.output(output)
        return output #- torch.mean(output, dim=1).unsqueeze(-1)


    


class TCN_Block(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation, padding):
        super().__init__()
        self.padding = padding

        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_channels=input_dim, out_channels=output_dim,
                                                     kernel_size=kernel_size, stride=1, dilation=dilation, padding=padding))
        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(in_channels=output_dim, out_channels=output_dim,
                                                     kernel_size=kernel_size, stride=1, dilation=dilation, padding=padding))
        self.dropout2 = nn.Dropout(p=0.2)

        self.downsample = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=1)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        # Do the first dialated conv
        tcn_output = self.conv1(x)
        # remove excess padding
        tcn_output = tcn_output[:, :, :-self.padding]
        tcn_output = self.leaky(tcn_output)
        tcn_output = self.dropout1(tcn_output)

        # do the second dilated conv
        tcn_output = self.conv2(tcn_output)
        tcn_output = tcn_output[:, :, :-self.padding]
        tcn_output = self.leaky(tcn_output)
        tcn_output = self.dropout2(tcn_output)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(tcn_output + x)
