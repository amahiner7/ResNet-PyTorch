from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, params):
        super().__init__()

        self.out_channels_list = params['out_channels']
        self.kernel_size_list = params['kernel_size']
        self.stride_list = params['stride']
        self.padding_list = params['padding']
        self.num_blocks = params['num_blocks']
        self.out_channels = in_channels

        self.block_list = nn.ModuleList()

        for index in range(self.num_blocks):
            block = self._create_block(index)
            self.block_list.append(block)

        self.shortcut_conv = nn.Conv2d(in_channels=in_channels,
                                       out_channels=self.out_channels_list[2],
                                       kernel_size=self.kernel_size_list[0],
                                       stride=self.stride_list[0][0],
                                       padding=self.padding_list[0],
                                       bias=False)

        self.shortcut_batch_norm = nn.BatchNorm2d(num_features=self.out_channels_list[2])

        self.activation = nn.ReLU()

    def _create_block(self, index):
        block = nn.ModuleList()

        if index == 0:
            block.append(self._create_Conv2d(out_channels=self.out_channels_list[0],
                                             kernel_size=self.kernel_size_list[0],
                                             stride=self.stride_list[0][0],
                                             padding=self.padding_list[0]))
        else:
            block.append(self._create_Conv2d(out_channels=self.out_channels_list[0],
                                             kernel_size=self.kernel_size_list[0],
                                             stride=self.stride_list[0][1],
                                             padding=self.padding_list[0]))

        block.append(nn.BatchNorm2d(num_features=self.out_channels))
        block.append(nn.ReLU())

        block.append(self._create_Conv2d(out_channels=self.out_channels_list[1],
                                         kernel_size=self.kernel_size_list[1],
                                         stride=self.stride_list[1],
                                         padding=self.padding_list[1]))
        block.append(nn.BatchNorm2d(num_features=self.out_channels))
        block.append(nn.ReLU())

        block.append(self._create_Conv2d(out_channels=self.out_channels_list[2],
                                         kernel_size=self.kernel_size_list[2],
                                         stride=self.stride_list[2],
                                         padding=self.padding_list[2]))
        block.append(nn.BatchNorm2d(num_features=self.out_channels))

        return block

    def _create_Conv2d(self, out_channels, kernel_size, stride, padding):
        conv2d = nn.Conv2d(in_channels=self.out_channels,
                           out_channels=out_channels,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           bias=False)

        self.out_channels = out_channels

        return conv2d

    def forward(self, inputs):
        shortcut = inputs
        x = inputs

        for block_index in range(len(self.block_list)):
            block = self.block_list[block_index]

            for layer_index in range(len(block)):
                layer = block[layer_index]
                x = layer(x)

            if block_index == 0:
                shortcut = self.shortcut_conv(shortcut)
                shortcut = self.shortcut_batch_norm(shortcut)

            x = x + shortcut
            x = self.activation(x)

            shortcut = x

        return x
