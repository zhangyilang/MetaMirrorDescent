import math
from string import ascii_uppercase
from collections import OrderedDict
from typing import Iterable, Union, Sequence, Optional, Callable

import torch
import torch.nn as nn

from src.meta_alg_base import MetaLearningAlgBase


class KronLinear(nn.Module):
    def __init__(self, in_size: Sequence[int], out_size: Sequence[int], bias: bool = True,
                 device: Optional[Union[torch.device, str]] = None, dtype: Optional[type] = None,
                 zero_init: bool = False) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if not isinstance(in_size, torch.Size):
            in_size = torch.Size(in_size)
        self.in_size = in_size
        if not isinstance(out_size, torch.Size):
            out_size = torch.Size(out_size)
        self.out_size = out_size
        self.zero_init = zero_init

        self.weights = nn.ParameterList()
        for out_features, in_features in zip(out_size, in_size):
            self.weights.append(nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs)))

        if bias:
            self.bias = nn.Parameter(torch.empty(*out_size, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_weight_ = nn.init.zeros_ if self.zero_init \
            else lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        for weight in self.weights:
            init_weight_(weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def format_einsum_str(self, dim: int) -> str:
        prefix = ascii_uppercase[:dim]
        suffix = ascii_uppercase[dim+1:len(self.in_size)]
        return f'ij,{prefix}j{suffix}->{prefix}i{suffix}'

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        assert input_tensor.dim() == len(self.in_size)

        for dim, weight in enumerate(self.weights):
            input_tensor = torch.einsum(self.format_einsum_str(dim), weight, input_tensor)
        output_tensor = input_tensor + self.bias if self.bias is not None else input_tensor

        return output_tensor

    def extra_repr(self) -> str:
        return 'in_size={}, out_size={}, bias={}'.format(
            self.in_size, self.out_size, self.bias is not None
        )


class HadamardLinear(nn.Module):
    def __init__(self, in_features: int, bias: bool = True,
                 device: Optional[Union[torch.device, str]] = None, dtype: Optional[type] = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features

        self.log_weight = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.log_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor * self.log_weight.exp() + self.bias

    def extra_repr(self) -> str:
        return 'in_features=out_features={}, bias={}'.format(
            self.in_features, self.bias is not None
        )


class KronMLPEncoder(nn.Module):
    def __init__(self, in_size: Sequence[int], hidden_shrink: Sequence[int],
                 activation: Callable = nn.ReLU) -> None:
        super().__init__()

        self.mlp = list()
        if not isinstance(in_size, torch.Tensor):
            in_size = torch.tensor(in_size)
        if in_size.numel() >= 4:    # Conv2d().weight
            in_size = torch.cat((in_size[:2], in_size[2:].prod(dim=0, keepdim=True)))
            self.mlp.append(nn.Flatten(start_dim=2))

        for shrink in hidden_shrink:
            out_size = torch.maximum(in_size.div(shrink, rounding_mode='floor'), torch.tensor(1))
            self.mlp.append(KronLinear(in_size, out_size))
            self.mlp.append(activation())
            in_size = out_size

        self.mlp.append(nn.Flatten(start_dim=0))
        self.mlp = nn.Sequential(*self.mlp)
        self.out_features = out_size.prod()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.mlp(input_tensor)


class KronMLPDecoder(nn.Module):
    def __init__(self, out_size: Sequence[int], hidden_shrink: Sequence[int],
                 activation: Callable = nn.ReLU) -> None:
        super().__init__()

        self.mlp = list()
        self.log_scale = list()
        self.shift = list()
        if not isinstance(out_size, torch.Tensor):
            out_size = torch.tensor(out_size)
        if out_size.numel() >= 4:    # Conv2d().weight
            unflattened_size = out_size[2:].tolist()
            self.log_scale.insert(0, nn.Unflatten(2, unflattened_size))
            self.shift.insert(0, nn.Unflatten(2, unflattened_size))
            out_size = torch.cat((out_size[:2], out_size[2:].prod(dim=0, keepdim=True)), dim=0)

        in_size = torch.maximum(out_size.div(hidden_shrink[0], rounding_mode='floor'), torch.tensor(1))
        self.log_scale.insert(0, KronLinear(in_size, out_size, zero_init=True))
        self.shift.insert(0, KronLinear(in_size, out_size, zero_init=True))

        self.mlp.insert(0, activation())
        for shrink in hidden_shrink[1:]:
            out_size = in_size
            in_size = torch.maximum(out_size.div(shrink, rounding_mode='floor'), torch.tensor(1))
            self.mlp.insert(0, KronLinear(in_size, out_size))
            self.mlp.insert(0, activation())

        self.mlp = nn.Sequential(*self.mlp)
        self.log_scale = nn.Sequential(*self.log_scale)
        self.shift = nn.Sequential(*self.shift)
        self.in_features = in_size.prod()
        self.in_size = torch.Size(in_size)

    def forward(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.mlp(input_tensor)
        return self.log_scale(features).exp(), self.shift(features)


class LayerIAF(nn.Module):
    def __init__(self, named_params: Iterable[tuple[str, torch.Tensor]],
                 embed_shrink: Sequence[int] = (2, 2, 2)) -> None:
        super().__init__()
        names, params = tuple(zip(*named_params))
        self.layers = tuple(self._name2layer(name) for name in names)

        # No encoder for the last layer
        self.encoders = nn.ModuleDict()
        for layer, param in zip(self.layers[:-1], params[:-1]):
            self.encoders[layer] = KronMLPEncoder(param.size(), embed_shrink)

        # No decoder for the first layer
        self.decoders = nn.ModuleDict()
        for layer, param in zip(self.layers[1:], params[1:]):
            self.decoders[layer] = KronMLPDecoder(param.size(), embed_shrink)

        # Layer-wise inverse autoregressor
        self.IARs = nn.ModuleDict({self.layers[0]: HadamardLinear(params[0].numel())})
        cumulative_features = self.encoders[self.layers[0]].out_features.item()
        for layer in self.layers[1:]:
            layer_features = self.decoders[layer].in_features
            self.IARs[layer] = nn.Linear(cumulative_features, layer_features)
            cumulative_features += layer_features.item()

    @staticmethod
    def _name2layer(name: str) -> str:
        return name.replace('.', '_')

    @staticmethod
    def _layer2name(layer: str) -> str:
        return layer.replace('_', '.')

    def forward(self, input_named_tensors: Union[Iterable[tuple[str, torch.Tensor]],
    OrderedDict[str, torch.Tensor]]) -> OrderedDict[str, torch.Tensor]:
        if not isinstance(input_named_tensors, OrderedDict):
            input_named_tensors = OrderedDict(input_named_tensors)
        output_named_tensors = OrderedDict()

        # encode
        enc_embeds = OrderedDict()
        for layer in self.layers[:-1]:
            input_tensor = input_named_tensors[self._layer2name(layer)]
            enc_embeds[layer] = self.encoders[layer](input_tensor)

        # decode first block
        layer = self.layers[0]
        input_tensor = input_named_tensors[self._layer2name(layer)]
        dec_embed = self.IARs[layer](input_tensor.flatten())
        output_named_tensors[self._layer2name(layer)] = dec_embed.view_as(input_tensor)

        # decode other blocks
        cumulative_embed = list()
        last_layer = layer
        for layer in self.layers[1:]:
            cumulative_embed.append(enc_embeds[last_layer])
            name = self._layer2name(layer)
            dec_embed = self.IARs[layer](torch.cat(cumulative_embed))
            decoder = self.decoders[layer]
            scale, shift = decoder(dec_embed.view(decoder.in_size))
            output_named_tensors[name] = input_named_tensors[name] * scale + shift
            last_layer = layer

        return output_named_tensors


class BlockIAF(nn.Module):
    def __init__(self, named_params: Iterable[tuple[str, torch.Tensor]],
                 num_layers_per_blk: Union[int, Iterable] = 4,
                 embed_shrink: Sequence[int] = (2, 2, 2)) -> None:
        super().__init__()
        names, params = tuple(zip(*named_params))
        all_layers = tuple(self._name2layer(name) for name in names)

        self._layers_per_blk = list()
        if isinstance(num_layers_per_blk, int):
            div, mod = divmod(len(all_layers),  num_layers_per_blk)
            num_layers_per_blk = [num_layers_per_blk, ] * div
            if mod != 0:
                num_layers_per_blk += [mod]
        else:
            assert sum(num_layers_per_blk) == len(all_layers)
        pointer = 0
        for num_layers in num_layers_per_blk:
            self._layers_per_blk.append(all_layers[pointer:pointer + num_layers])
            pointer += num_layers

        # No encoder for the last block
        self.layer_encoders = nn.ModuleDict()
        for layer, param in zip(all_layers[:-num_layers_per_blk[-1]], params[:-num_layers_per_blk[-1]]):
            self.layer_encoders[layer] = KronMLPEncoder(param.size(), embed_shrink)

        # No decoder for the first block
        self.layer_decoders = nn.ModuleDict()
        for layer, param in zip(all_layers[num_layers_per_blk[0]:], params[num_layers_per_blk[0]:]):
            self.layer_decoders[layer] = KronMLPDecoder(param.size(), embed_shrink)

        # Block-wise inverse autoregressor
        blk_features = sum((param.numel() for param in params[:num_layers_per_blk[0]]))
        self.blk_IARs = nn.ModuleDict({self._layers2blk(self._layers_per_blk[0]):
                                             HadamardLinear(blk_features)})
        cumulative_features = sum((self.layer_encoders[layer].out_features
                                   for layer in self._layers_per_blk[0]))
        for layers in self._layers_per_blk[1:]:
            blk_features = sum((self.layer_decoders[layer].in_features for layer in layers))
            self.blk_IARs[self._layers2blk(layers)] = nn.Linear(cumulative_features, blk_features)
            cumulative_features += blk_features

    @staticmethod
    def _layers2blk(layers_blk: Sequence[str]) -> str:
        return '-'.join(layers_blk)

    @staticmethod
    def _name2layer(name: str) -> str:
        return name.replace('.', '_')

    @staticmethod
    def _layer2name(layer: str) -> str:
        return layer.replace('_', '.')

    def forward(self, input_named_tensors: Union[Iterable[tuple[str, torch.Tensor]],
                OrderedDict[str, torch.Tensor]]) -> OrderedDict[str, torch.Tensor]:
        if not isinstance(input_named_tensors, OrderedDict):
            input_named_tensors = OrderedDict(input_named_tensors)
        output_named_tensors = OrderedDict()

        # encode
        enc_blk_embeds = OrderedDict()
        for layers in self._layers_per_blk[:-1]:
            enc_blk_embed = []
            for layer in layers:
                input_tensor = input_named_tensors[self._layer2name(layer)]
                enc_blk_embed.append(self.layer_encoders[layer](input_tensor))
            enc_blk_embeds[self._layers2blk(layers)] = nn.utils.parameters_to_vector(enc_blk_embed)

        # decode first block
        layers = self._layers_per_blk[0]
        blk = self._layers2blk(layers)
        input_tensors_blk_vec = nn.utils.parameters_to_vector((input_named_tensors[self._layer2name(layer)]
                                                               for layer in layers))
        dec_blk_embed = self.blk_IARs[blk](input_tensors_blk_vec)
        pointer = 0
        for layer in layers:
            name = self._layer2name(layer)
            input_tensor = input_named_tensors[name]
            numel_embed = input_tensor.numel()
            output_named_tensors[name] = dec_blk_embed[pointer:pointer+numel_embed].view_as(input_tensor)
            pointer += numel_embed

        # decode other blocks
        cumulative_embed = list()
        for layers in self._layers_per_blk[1:]:
            cumulative_embed.append(enc_blk_embeds[blk])    # last blk
            blk = self._layers2blk(layers)                  # current blk
            dec_blk_embed = self.blk_IARs[blk](torch.cat(cumulative_embed))
            pointer = 0
            for layer in layers:
                decoder = self.layer_decoders[layer]
                numel_embed = decoder.in_features
                scale, shift = decoder(dec_blk_embed[pointer:pointer+numel_embed].view(decoder.in_size))
                name = self._layer2name(layer)
                output_named_tensors[name] = input_named_tensors[name] * scale + shift
                pointer += numel_embed

        return output_named_tensors


class MetaMirrorDescent(MetaLearningAlgBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    def _get_meta_model(self) -> dict[str, nn.Module]:
        return {'init': self._get_base_model(),
                'inverse_mirror_map': BlockIAF(self._base_model.meta_named_parameters())}

    def _get_meta_optimizer(self) -> Union[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        if self.args.dataset.lower() == 'omniglot':
            meta_optimizer = torch.optim.Adam([{'params': module.parameters()}
                                               for module in self._meta_model.values()],
                                              lr=self.args.meta_lr)
        else:   # diverge with Adam in some setups; use SGD instead for stability
            meta_optimizer = torch.optim.SGD([{'params': module.parameters()}
                                              for module in self._meta_model.values()],
                                             lr=self.args.meta_lr,
                                             momentum=0.9,
                                             weight_decay=1e-4,
                                             nesterov=True)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(meta_optimizer, step_size=20000)
    
        return meta_optimizer, lr_scheduler

    def adapt(self, trn_input: torch.Tensor, trn_target: torch.Tensor,
              first_order: bool = False) -> OrderedDict[str, nn.Parameter]:
        mirror_params = OrderedDict(self._meta_model['init'].named_parameters())
        params = self._meta_model['inverse_mirror_map'](mirror_params)

        for _ in range(self.args.task_iter):
            trn_logit = self._base_model(trn_input, params=params)
            task_nll = self._nll(trn_logit, trn_target)
            grads = torch.autograd.grad(task_nll,
                                        params.values(),
                                        create_graph=not first_order)

            for (name, mirror_param), grad in zip(mirror_params.items(), grads):
                mirror_params[name] = mirror_param - self.args.task_lr * grad
            params = self._meta_model['inverse_mirror_map'](mirror_params)

        return params


if __name__ == '__main__':
    from src.models import FourBlkCNN
    cnn = FourBlkCNN(num_classes=5)
    blk_IAF = BlockIAF(cnn.named_parameters())
    named_params_trans = blk_IAF(cnn.named_parameters())
    for name, param in cnn.named_parameters():
        print(name, named_params_trans[name].size(), (param - named_params_trans[name]).abs().mean().item())

    # print(sum(param.numel() for param in cnn.parameters()))
    # print(sum(param.numel() for param in blk_IAF.layer_encoders.parameters()))
    # print(sum(param.numel() for param in blk_IAF.layer_decoders.parameters()))
    # print(sum(param.numel() for param in blk_IAF.blk_IARs.parameters()))

