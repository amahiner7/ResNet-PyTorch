from torch import nn


class Conv1_Block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.out_channels = 64

        self.conv1_pad = nn.ZeroPad2d(padding=3)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=self.out_channels,
                               kernel_size=(7, 7),
                               stride=(2, 2),
                               padding=0,
                               bias=False)
        self.bn_conv1 = nn.BatchNorm2d(num_features=self.out_channels)
        self.activation_1 = nn.ReLU()
        self.pool1_pad = nn.ZeroPad2d(padding=1)

    def forward(self, inputs):
        x = self.conv1_pad(inputs)
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.activation_1(x)
        x = self.pool1_pad(x)

        return x
