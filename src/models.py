from collections import OrderedDict
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torchmeta.modules import MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d, MetaLinear


def _conv3x3(in_channels: int, out_channels: int, **kwargs) -> MetaModule:
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


def _conv3x3_strided(in_channels: int, out_channels: int, padding=1, stride: int = 2, **kwargs) -> MetaModule:
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=stride, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU()
    )


class FourBlkCNN(MetaModule):
    def __init__(self, num_classes: int, in_channels: int = 3, hidden_size: int = 64) -> None:
        super(FourBlkCNN, self).__init__()
        if isinstance(hidden_size, Sequence):
            hidden_size = tuple(hidden_size)
        elif isinstance(hidden_size, int):
            hidden_size = (hidden_size, ) * 4
        else:
            raise ValueError

        self.features = MetaSequential(
            _conv3x3(in_channels, hidden_size[0]),
            _conv3x3(hidden_size[0], hidden_size[1]),
            _conv3x3(hidden_size[1], hidden_size[2]),
            _conv3x3(hidden_size[2], hidden_size[3]),
            nn.Flatten()
        )
        self.classifier = MetaLinear(25*hidden_size[3], num_classes)

    def forward(self, inputs: torch.Tensor, params: Optional[OrderedDict[str, nn.Parameter]] = None) -> torch.Tensor:
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))

        return logits


class FourBlkCNNOmniglot(MetaModule):
    def __init__(self, num_classes: int, in_channels: int = 1, hidden_size: int = 64) -> None:
        super(FourBlkCNNOmniglot, self).__init__()
        if isinstance(hidden_size, Sequence):
            hidden_size = tuple(hidden_size)
        elif isinstance(hidden_size, int):
            hidden_size = (hidden_size, ) * 4
        else:
            raise ValueError

        self.features = MetaSequential(
            # _conv3x3_strided(in_channels, hidden_size[0], padding=0),
            # _conv3x3_strided(hidden_size[0], hidden_size[1], padding=0),
            # _conv3x3_strided(hidden_size[1], hidden_size[2], padding=0),
            # _conv3x3_strided(hidden_size[2], hidden_size[3], padding=1),
            _conv3x3(in_channels, hidden_size[0]),
            _conv3x3(hidden_size[0], hidden_size[1]),
            _conv3x3(hidden_size[1], hidden_size[2]),
            _conv3x3(hidden_size[2], hidden_size[3]),
            nn.Flatten()
        )
        self.classifier = MetaLinear(hidden_size[3], num_classes)

    def forward(self, inputs: torch.Tensor, params: Optional[OrderedDict[str, nn.Parameter]] = None) -> torch.Tensor:
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))

        return logits


def _res_conv3x3(in_channels: int, out_channels: int, **kwargs) -> MetaModule:
    return MetaSequential(
        MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        MetaBatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.LeakyReLU(0.1)
    )


class _ResBlk(MetaModule):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(_ResBlk, self).__init__()
        self.conv = MetaSequential(
            _res_conv3x3(in_channels, out_channels),
            _res_conv3x3(out_channels, out_channels),
            _res_conv3x3(out_channels, out_channels),
        )
        self.shortcut = MetaConv2d(in_channels, out_channels, kernel_size=1)
        self.pooling = nn.MaxPool2d(2, ceil_mode=True)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs: torch.Tensor, params: Optional[OrderedDict[str, nn.Parameter]] = None) -> torch.Tensor:
        conv_features = self.conv(inputs, params=self.get_subdict(params, 'conv'))
        res_features = self.shortcut(inputs, params=self.get_subdict(params, 'shortcut'))
        outputs = self.dropout(self.pooling(conv_features + res_features))

        return outputs


class SimpleResNet12(MetaModule):
    # ResNet12 from SNAIL (Mishra et al.'18), which has fewer channels and
    # simpler structure compared to the one used in MetaOptNet (Lee et al.'18)
    def __init__(self, num_classes: int, in_channels: int = 3,
                 hidden_size: Union[int, Sequence[int]] = (64, 96, 128, 256),
                 num_feat: int = 384) -> None:
        super(SimpleResNet12, self).__init__()
        if isinstance(hidden_size, Sequence):
            hidden_size = tuple(hidden_size)
        elif isinstance(hidden_size, int):
            hidden_size = (hidden_size, ) * 4
        else:
            raise ValueError

        self.features = MetaSequential(
            _ResBlk(in_channels, hidden_size[0]),
            _ResBlk(hidden_size[0], hidden_size[1]),
            _ResBlk(hidden_size[1], hidden_size[2]),
            _ResBlk(hidden_size[2], hidden_size[3]),
            MetaConv2d(hidden_size[3], 2048, kernel_size=1),
            nn.AvgPool2d(6),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            MetaConv2d(2048, 384, kernel_size=1),
            nn.Flatten()
        )

        self.classifier = MetaLinear(num_feat, num_classes)

    def forward(self, inputs: torch.Tensor, params: Optional[OrderedDict[str, nn.Parameter]] = None) -> torch.Tensor:
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))

        return logits


if __name__ == '__main__':
    cnn = FourBlkCNN(5, 3, 128)
    print(nn.utils.parameters_to_vector(cnn.parameters()).numel())
