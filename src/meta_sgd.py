from typing import Iterable
from collections import OrderedDict

import torch
import torch.nn as nn
from torchmeta.utils import gradient_update_parameters

from src.meta_alg_base import MetaLearningAlgBase


def _log_lr(named_params: Iterable) -> nn.Module:
    log_lr = nn.ParameterDict()

    for name, param in named_params:
        log_lr[name.replace('.', '_')] = nn.Parameter(torch.zeros_like(param))

    return log_lr


class MetaSGD(MetaLearningAlgBase):
    def __init__(self, args):
        super().__init__(args)

    def _get_meta_model(self) -> dict[str, nn.Module]:
        return {'init': self._get_base_model(),
                'log_lr': _log_lr(self._base_model.named_parameters())}

    def adapt(self, trn_input: torch.Tensor, trn_target: torch.Tensor,
              first_order: bool = False) -> OrderedDict[str, nn.Parameter]:
        params = OrderedDict(self._meta_model['init'].named_parameters())
        task_lr = OrderedDict({name.replace('_', '.'): log_lr.exp() * self.args.task_lr
                               for name, log_lr in self._meta_model['log_lr'].items()})

        for _ in range(self.args.task_iter):
            trn_logit = self._base_model(trn_input, params=params)
            task_loss = self._nll(trn_logit, trn_target)
            params = gradient_update_parameters(self._base_model,
                                                task_loss,
                                                params=params,
                                                step_size=task_lr,
                                                first_order=first_order)

        return params
