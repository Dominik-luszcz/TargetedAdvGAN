import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, output_dim, sample_size):
        super().__init__()

        self.epsilon = 0.2

        self.output_dim = output_dim
        self.sample_size = sample_size


        #self.gru = nn.GRU(input_size=2, hidden_size=128, num_layers=4, batch_first=True)

        self.generator = nn.Sequential(
            TCN_Block(input_dim=1, output_dim=64, kernel_size=3, dilation=1, padding=(3-1) * 1),
            TCN_Block(input_dim=64, output_dim=128, kernel_size=5, dilation=2, padding=(5-1) * 2),
            TCN_Block(input_dim=128, output_dim=64, kernel_size=5, dilation=4, padding=(5-1) * 4),
            TCN_Block(input_dim=64, output_dim=32, kernel_size=3, dilation=8, padding=(3-1) * 8),
        )
        self.output = nn.Linear(32, 1)

        # self.generator = nn.Sequential(
        #     TCN_Block(input_dim=1, output_dim=64, kernel_size=3, dilation=1, padding=(3-1) * 1),
        #     TCN_Block(input_dim=64, output_dim=128, kernel_size=5, dilation=2, padding=(5-1) * 2),
        #     TCN_Block(input_dim=128, output_dim=64, kernel_size=5, dilation=4, padding=(5-1) * 4),
        #     TCN_Block(input_dim=64, output_dim=32, kernel_size=3, dilation=8, padding=(3-1) * 8),
        # )
        # self.output = nn.Linear(32, 1)

        #self.output = nn.Linear(128, 1)


        # self.generator = nn.Sequential(
        #     nn.ConvTranspose1d(in_channels=1, out_channels=128, kernel_size=5, stride=2), # 4 for full recording (2, 2, 2) for random 500 sample
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Dropout(0.25),

        #     nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=5, stride=2), # 3
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Dropout(0.25),

        #     nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=5, stride=2), # 4
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Dropout(0.25),

        #     nn.Conv1d(in_channels=128, out_channels=self.output_dim, kernel_size=5, stride=1, padding=5//2),
        #     nn.Tanh() # force returns between -1 and 1, then after multiply by epsilon to ensure its reasonable (between -0.2 and 0.2)
        # )


        self.double()

    def forward(self, noise):
        #noise = torch.concatenate([condition, z], dim=1)
        noise = torch.permute(noise, (0, 2, 1))
        output = self.generator(noise)
        output = torch.permute(output, (0, 2, 1))
        #output, _ = self.gru(noise)
        #output = output[:, :, :self.sample_size] # will not hardcode it, fix this after
        #output = output[:, :, :2433]
        #output = output - torch.mean(output)
        #output = 0.1 * output # enforce returns to be between -0.2 and 0.2
        return self.output(output)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # self.gru = nn.GRU(input_size=2, hidden_size=128, num_layers=5, batch_first=True)
        # self.output = nn.Linear(128, output_dim)

        # self.discriminator = nn.Sequential(
        #     TCN_Block(input_dim=input_dim, output_dim=hidden_dim, kernel_size=5, dilation=1, padding=(5-1) * 1), # padding (kernel_size - 1) * dilation
        #     TCN_Block(input_dim=hidden_dim, output_dim=hidden_dim*2, kernel_size=5, dilation=2, padding=(5-1) * 2),
        #     TCN_Block(input_dim=hidden_dim*2, output_dim=hidden_dim*3, kernel_size=5, dilation=4, padding=(5-1) * 4),
        #     TCN_Block(input_dim=hidden_dim*3, output_dim=hidden_dim*2, kernel_size=5, dilation=8, padding=(5-1) * 8),
        #     TCN_Block(input_dim=hidden_dim*2, output_dim=hidden_dim, kernel_size=5, dilation=16, padding=(5-1) * 16),

        #     nn.AdaptiveAvgPool1d(2),
        #     nn.Flatten(),

        #     nn.Linear(hidden_dim*2, hidden_dim),
        #     nn.LeakyReLU(0.02),
        #     nn.Linear(hidden_dim, output_dim)
        # )

        self.tcn1 = TCN_Block(input_dim=1, output_dim=64, kernel_size=3, dilation=1, padding=(3-1) * 1)
        self.gru1 = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.leaky = nn.LeakyReLU()
        self.tcn2 = TCN_Block(input_dim=128, output_dim=128, kernel_size=5, dilation=2, padding=(5-1) * 2)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        self.tcn3 = TCN_Block(input_dim=64, output_dim=32, kernel_size=5, dilation=4, padding=(5-1) * 4)
        self.mlp = nn.Sequential(
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 1)
        )

        # self.tcn1 = TCN_Block(input_dim=1, output_dim=64, kernel_size=3, dilation=1, padding=(3-1) * 1)
        # self.gru1 = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        # self.leaky = nn.LeakyReLU()
        # self.tcn2 = TCN_Block(input_dim=128, output_dim=128, kernel_size=5, dilation=2, padding=(5-1) * 2)
        # self.gru2 = nn.GRU(input_size=128, hidden_size=64, num_layers=1, batch_first=True)
        # self.tcn3 = TCN_Block(input_dim=64, output_dim=32, kernel_size=5, dilation=4, padding=(5-1) * 4)
        # self.mlp = nn.Sequential(
        #     nn.Linear(64, 32),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(32, 1)
        # )

        self.double()

    def forward(self, x):
        
        #x = torch.concatenate([condition, data], dim=1) # align on  seq_len

        x = torch.permute(x, (0, 2, 1))
        x = self.tcn1(x)
        x = torch.permute(x, (0, 2, 1))
        with torch.backends.cudnn.flags(enabled=False):
            x, _ = self.gru1(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.tcn2(x)
        x = torch.permute(x, (0, 2, 1))
        with torch.backends.cudnn.flags(enabled=False):
            x, _ = self.gru2(x)
        x = torch.permute(x, (0, 2, 1))
        self.tcn3(x)
        x = torch.permute(x, (0, 2, 1))
        x = x.mean(dim=1)
        return self.mlp(x)
        


        #x = torch.permute(x, (0, 2, 1))
        # with torch.backends.cudnn.flags(enabled=False):
        #     gru_output, _ = self.gru(x)
        # output = self.output(gru_output)
        # return output.mean(dim=1)

        # x = torch.permute(x, (0, 2, 1))
        # output = self.discriminator(x)
        # output = torch.permute(output, (0, 2, 1))
        # return output.mean(dim=1)
    


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
