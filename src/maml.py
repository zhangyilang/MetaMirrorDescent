from collections import OrderedDict

import torch
import torch.nn as nn
from torchmeta.utils import gradient_update_parameters

from src.meta_alg_base import MetaLearningAlgBase


class MAML(MetaLearningAlgBase):
    def __init__(self, args) -> None:
        super(MAML, self).__init__(args)

    def _get_meta_model(self) -> dict[str, nn.Module]:
        return {'init': self._get_base_model()}

    def adapt(self, trn_input: torch.Tensor, trn_target: torch.Tensor,
              first_order: bool = False) -> OrderedDict[str, nn.Parameter]:
        params = OrderedDict(self._meta_model['init'].named_parameters())

        for _ in range(self.args.task_iter):
            trn_logit = self._base_model(trn_input, params=params)
            task_loss = self._nll(trn_logit, trn_target)
            params = gradient_update_parameters(self._base_model,
                                                task_loss,
                                                params=params,
                                                step_size=self.args.task_lr,
                                                first_order=first_order)

        return params
