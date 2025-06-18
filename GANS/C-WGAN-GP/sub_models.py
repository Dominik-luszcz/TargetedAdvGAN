import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, output_dim, sample_size):
        super().__init__()

        self.epsilon = 0.2

        self.output_dim = output_dim
        self.sample_size = sample_size

        self.generator = nn.Sequential(

            TCN_Block(input_dim=2, output_dim=128, kernel_size=5, dilation=1, padding=(5-1) * 1),

            TCN_Block(input_dim=128, output_dim=128, kernel_size=5, dilation=2, padding=(5-1) * 2),

            TCN_Block(input_dim=128, output_dim=128, kernel_size=5, dilation=4, padding=(5-1) * 4),

            TCN_Block(input_dim=128, output_dim=64, kernel_size=5, dilation=8, padding=(5-1) * 8),
            #nn.Tanh()
        )

        self.output = nn.Linear(64, 1)


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

    def forward(self, condition, z):
        noise = torch.concatenate([condition, z], dim=-1)
        noise = torch.permute(noise, (0, 2, 1))
        output = self.generator(noise)
        #output = output[:, :, :self.sample_size] # will not hardcode it, fix this after
        #output = output[:, :, :2433]
        #output = output - torch.mean(output)
        #output = 0.1 * output # enforce returns to be between -0.2 and 0.2
        return self.output(torch.permute(output, (0,2,1)))

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.discriminator = nn.Sequential(
            TCN_Block(input_dim=input_dim, output_dim=hidden_dim, kernel_size=5, dilation=1, padding=(5-1) * 1), # padding (kernel_size - 1) * dilation
            TCN_Block(input_dim=hidden_dim, output_dim=hidden_dim*2, kernel_size=5, dilation=2, padding=(5-1) * 2),
            TCN_Block(input_dim=hidden_dim*2, output_dim=hidden_dim*3, kernel_size=5, dilation=4, padding=(5-1) * 4),
            TCN_Block(input_dim=hidden_dim*3, output_dim=hidden_dim*2, kernel_size=5, dilation=8, padding=(5-1) * 8),
            TCN_Block(input_dim=hidden_dim*2, output_dim=hidden_dim, kernel_size=5, dilation=16, padding=(5-1) * 16),

            nn.AdaptiveAvgPool1d(2),
            nn.Flatten(),

            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(0.02),
            nn.Linear(hidden_dim, output_dim)
        )



        # self.discriminator = nn.Sequential(
        #     nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=5, stride=1, padding=5//2),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Dropout(0.25),

        #     nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=5//2),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Dropout(0.25),

        #     nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=5//2),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Dropout(0.25),

        #     nn.Conv1d(in_channels=128, out_channels=1, kernel_size=5, stride=1, padding=5//2),
        #     #nn.Sigmoid()
        # )

        # self.discriminator = nn.Sequential(
        #     nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_dim, kernel_size=5, stride=2, padding=1),
        #     nn.BatchNorm1d(self.hidden_dim),
        #     nn.LeakyReLU(negative_slope=0.2),

        #     nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim * 2, kernel_size=5, stride=2, padding=1),
        #     nn.BatchNorm1d(self.hidden_dim * 2),
        #     nn.LeakyReLU(negative_slope=0.2),

        #     nn.Conv1d(in_channels=self.hidden_dim * 2, out_channels=self.hidden_dim, kernel_size=5, stride=2, padding=1),
        #     nn.BatchNorm1d(self.hidden_dim),
        #     nn.LeakyReLU(negative_slope=0.2),

        #     # By here we will have a tensor of size [B, self.hidden_dim, seq_len]
        #     # But seq len could be somewhat variable depending on how long of a time series we want to generate
        #     # So instead of standard AvgPooling where we need to know the seq_len, do an adaptive avg pool to get seq_len of self.hidden_dim * output argmument (4 in this case)
        #     nn.AdaptiveAvgPool1d(4),

        #     nn.Flatten(),
        #     nn.Linear(in_features=self.hidden_dim * 4, out_features=self.output_dim),
        #     nn.Sigmoid()

        # )

        self.double()

    def forward(self, condition, data):
        
        x = torch.concatenate([condition, data], dim=-1)
        x = torch.permute(x, (0, 2, 1))
        tcn_output = self.discriminator(x)
        return tcn_output #.mean(dim=1)

        x = torch.permute(x, (0, 2, 1))
        output = self.discriminator(x)
        output = torch.permute(output, (0, 2, 1))
        return output.mean(dim=1)
    


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
