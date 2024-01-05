from torch import nn
import math


class ConvNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        h_in: int = 70,
        w_in: int = 140,

        kernel_size1: int = 5,
        stride1: int = 1,
        padding1: int = 2,
        conv_out1: int = 256,   

        kernel_size2: int = 3,
        stride2: int = 1,
        padding2: int = 1,
        conv_out2: int = 128,

        kernel_size3: int = 3,
        stride3: int = 1,
        padding3: int = 1,
        conv_out3: int = 64,

        kernel_size4: int = 3,
        stride4: int = 1,
        padding4: int = 1,
        conv_out4: int = 1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.h_in = h_in
        self.w_in = w_in

        # compute output size of conv2d w.r.t. pytorch documentation
        h_out1 = math.floor((h_in+2*padding1-kernel_size1)/stride1+1)
        w_out1 = math.floor((w_in+2*padding1-kernel_size1)/stride1+1)
        h_out2 = math.floor((h_out1+2*padding2-kernel_size2)/stride2+1)
        w_out2 = math.floor((w_out1+2*padding2-kernel_size2)/stride2+1) 
        h_out3 = math.floor((h_out2+2*padding3-kernel_size3)/stride3+1)
        w_out3 = math.floor((w_out2+2*padding3-kernel_size3)/stride3+1)

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels,
                conv_out1,
                kernel_size=kernel_size1,
                stride=stride1,
                padding=padding1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_out1,
                conv_out2,
                kernel_size=kernel_size2,
                stride=stride2,
                padding=padding2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_out2,
                conv_out3,
                kernel_size=kernel_size3,
                stride=stride3,
                padding=padding3,
            ),
            nn.ReLU(),
            nn.Conv2d(
                conv_out3,
                conv_out4,
                kernel_size=kernel_size4,
                stride=stride4,
                padding=padding4,
            )
        )

    def forward(self, x):
        batch_size, height, width, channels = x.size()

        # reshape to (batch_size, channels, height, width)
        x = x.reshape(batch_size, channels, height, width)

        return self.model(x)


if __name__ == "__main__":
    _ = ConvNet()