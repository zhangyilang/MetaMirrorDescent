from typing import Sequence, Union

import torch


def split_tasks(task_batch: dict[str, Sequence[torch.Tensor]],
                device: Union[torch.device, str]) -> Sequence[torch.Tensor]:
    x_supp_batch, y_supp_batch = task_batch['train']
    x_qry_batch, y_qry_batch = task_batch['test']

    x_supp_batch = x_supp_batch.to(device=device)
    y_supp_batch = y_supp_batch.to(device=device)
    x_qry_batch = x_qry_batch.to(device=device)
    y_qry_batch = y_qry_batch.to(device=device)

    return x_supp_batch, y_supp_batch, x_qry_batch, y_qry_batch


class Checkpointer:
    def __init__(self, save_fn: callable, alg_name: str) -> None:
        self.save_fn = save_fn
        self.alg_name = alg_name
        self.counter = 0
        self.best_acc = 0

    def update(self, acc: float) -> None:
        self.counter += 1
        self.save_fn(self.alg_name + '_{0:02d}.ct'.format(self.counter))

        if acc > self.best_acc:
            self.best_acc = acc
            self.save_fn(self.alg_name + '_final.ct'.format(self.counter))
