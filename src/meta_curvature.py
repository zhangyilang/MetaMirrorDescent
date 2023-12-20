from typing import Iterable
from collections import OrderedDict

import torch
import torch.nn as nn

from src.meta_alg_base import MetaLearningAlgBase


class LayerWiseKronLinear(nn.Module):
    def __init__(self, params: Iterable[torch.Tensor]) -> None:
        super().__init__()
        self.mc_weight = nn.ParameterList()

        for param in params:
            param_size = param.size()
            param_dim = param.dim()

            if param_dim == 1:      # Conv2d().bias / Linear().bias / BN
                self.mc_weight.append(nn.Parameter(torch.ones_like(param)))
            else:                   # Linear().weight / Conv2d().weight
                self.mc_weight.append(nn.Parameter(torch.eye(param_size[0])))
                self.mc_weight.append(nn.Parameter(torch.eye(param_size[1])))
                if param_dim == 4:  # Conv2d().weight
                    self.mc_weight.append(nn.Parameter(torch.eye(param_size[2] * param_size[3])))

    def forward(self, input_grads: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        output_grads = list()
        pointer = 0

        for input_grad in input_grads:
            param_dim = input_grad.dim()

            if param_dim == 1:  # Conv2d().bias / Linear().bias
                output_grad = self.mc_weight[pointer] * input_grad
                pointer += 1
            elif param_dim == 2:  # Linear().weight
                output_grad = self.mc_weight[pointer] @ input_grad @ self.mc_weight[pointer+1]
                pointer += 2
            elif param_dim == 4:  # Conv2d().weight
                output_grad = torch.einsum('ijk,il->ljk',
                                           input_grad.flatten(start_dim=2),
                                           self.mc_weight[pointer])
                output_grad = self.mc_weight[pointer+1] @ output_grad @ self.mc_weight[pointer+2]
                pointer += 3
            else:
                raise NotImplementedError

            output_grads.append(output_grad.view_as(input_grad))

        return output_grads


class MetaCurvature(MetaLearningAlgBase):
    def __init__(self, args) -> None:
        super().__init__(args)

    def _get_meta_model(self) -> dict[str, nn.Module]:
        return {'init': self._get_base_model(),
                'MC': LayerWiseKronLinear(self._base_model.parameters())}

    def adapt(self, trn_input: torch.Tensor, trn_target: torch.Tensor,
              first_order: bool = False) -> OrderedDict[str, nn.Parameter]:
        params = OrderedDict(self._meta_model['init'].named_parameters())
        for _ in range(self.args.task_iter):
            trn_logit = self._base_model(trn_input, params=params)
            task_nll = self._nll(trn_logit, trn_target)
            grads = torch.autograd.grad(task_nll,
                                        params.values(),
                                        create_graph=not first_order)
            grads = self._meta_model['MC'](grads)

            for (name, param), grad in zip(params.items(), grads):
                params[name] = param - self.args.task_lr * grad

        return params
