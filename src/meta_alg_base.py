import os
from typing import Union, Sequence, Optional
from collections import OrderedDict
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, Compose, Resize, InterpolationMode
from torchmeta.modules import MetaModule
from torchmeta.datasets import Omniglot, MiniImagenet, TieredImagenet, CUB, CIFARFS
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import ClassSplitter, Categorical, Rotation

from src.models import FourBlkCNN, FourBlkCNNOmniglot, SimpleResNet12
from src.utils import split_tasks, Checkpointer

_META_DATASETS = {'omniglot': Omniglot,
                  'miniimagenet': MiniImagenet,
                  'tieredimagenet': TieredImagenet,
                  'cub': CUB,
                  'cifarfs': CIFARFS}

_BASE_MODELS = {'cnn4': FourBlkCNN,
                'cnn4omni': FourBlkCNNOmniglot,
                'simple_resnet12': SimpleResNet12}


class MetaLearningAlgBase(ABC):
    @abstractmethod
    def __init__(self, args) -> None:
        self.args = args

        self.train_dataloader, self.val_dataloader, self.test_dataloader = self._get_meta_datasets()
        self._base_model = self._get_base_model()
        self._meta_model = self._get_meta_model()
        self._base_model.to('meta')
        for meta_module in self._meta_model.values():
            meta_module.to(self.args.device)
        self._nll = nn.CrossEntropyLoss()

    def _get_meta_datasets(self) -> tuple[BatchMetaDataLoader, BatchMetaDataLoader, BatchMetaDataLoader]:
        if self.args.dataset.lower() == 'omniglot':
            transform = Compose([Resize([28, 28]), ToTensor()])
            class_aug = [Rotation([90, 180, 270])]
        elif self.args.dataset.lower() == 'cub':
            transform = Compose([Resize([84, 84]), ToTensor()])
            class_aug = None
        else:
            transform = ToTensor()
            class_aug = None

        class_splitter_train = ClassSplitter(shuffle=True,
                                             num_train_per_class=self.args.num_supp,
                                             num_test_per_class=self.args.num_qry)
        class_splitter_eval = ClassSplitter(shuffle=True,
                                            num_train_per_class=self.args.num_supp,
                                            num_test_per_class=self.args.num_supp)

        kwargs = {'root': self.args.data_dir,
                  'num_classes_per_task': self.args.num_way,
                  'transform': transform,
                  'target_transform': Categorical(num_classes=self.args.num_way),
                  'class_augmentations': class_aug,
                  'download': self.args.download}
        dataset = _META_DATASETS[self.args.dataset.lower()]
        train_dataset = dataset(meta_train=True,
                                dataset_transform=class_splitter_train,
                                **kwargs)
        val_dataset = dataset(meta_val=True,
                              dataset_transform=class_splitter_eval,
                              **kwargs)
        test_dataset = dataset(meta_test=True,
                               dataset_transform=class_splitter_eval,
                               **kwargs)

        train_dataset.seed(self.args.seed)
        val_dataset.seed(self.args.seed)
        test_dataset.seed(self.args.seed)

        train_dataloader = BatchMetaDataLoader(train_dataset,
                                               batch_size=self.args.batch_size,
                                               num_workers=self.args.num_workers)
        val_dataloader = BatchMetaDataLoader(val_dataset, batch_size=1, num_workers=1)
        test_dataloader = BatchMetaDataLoader(test_dataset, batch_size=1, num_workers=1)

        return train_dataloader, val_dataloader, test_dataloader

    def _get_base_model(self) -> MetaModule:
        return _BASE_MODELS[self.args.base_model.lower()](self.args.num_way)

    @abstractmethod
    def _get_meta_model(self) -> dict[str, nn.Module]:
        raise NotImplementedError

    def _get_meta_optimizer(self) -> Union[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
        meta_optimizer = torch.optim.Adam([{'params': module.parameters()}
                                           for module in self._meta_model.values()],
                                          lr=self.args.meta_lr)

        return meta_optimizer, None

    @abstractmethod
    def adapt(self, trn_input: torch.Tensor, trn_target: torch.Tensor,
              first_order: bool = False) -> OrderedDict[str, nn.Parameter]:
        raise NotImplementedError

    def save_meta_model(self, file_name: str) -> None:
        torch.save({name: module.state_dict() for name, module in self._meta_model.items()},
                   os.path.join(self.args.model_dir, file_name))

    def load_meta_model(self, file_name: str) -> None:
        state_dicts = torch.load(os.path.join(self.args.model_dir, file_name))
        for name, module in self._meta_model.items():
            module.load_state_dict(state_dicts[name])

    def train(self) -> None:
        print('Training starts ...')
        meta_optimizer, lr_scheduler = self._get_meta_optimizer()
        check_pointer = Checkpointer(self.save_meta_model, self.args.algorithm.lower())

        running_loss = 0.
        running_acc = 0.
        self._base_model.train()

        # training loop
        for meta_idx, tasks in enumerate(self.train_dataloader):
            if meta_idx >= self.args.meta_iter:
                break

            for module in self._meta_model.values():
                module.train()
            meta_optimizer.zero_grad()

            for trn_input, trn_target, val_input, val_target in zip(*split_tasks(tasks, self.args.device)):
                params = self.adapt(trn_input, trn_target, first_order=self.args.first_order)
                val_logit = self._base_model(val_input, params=params)
                meta_loss = self._nll(val_logit, val_target) / self.args.batch_size
                meta_loss.backward()

                with torch.no_grad():
                    running_loss += meta_loss.detach().item()
                    running_acc += (val_logit.argmax(dim=1) == val_target).detach().float().mean().item()

            meta_optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            # meta-validation
            if (meta_idx + 1) % self.args.log_iter == 0:
                val_loss, val_acc = self.evaluate(self.val_dataloader, self.args.num_log_tasks)
                print('Meta-iter {0:d}: '
                      'train loss = {1:.3f}, train acc = {2:.2f}%, '
                      'val loss = {3:.3f}, val acc = {4:.1f}%'
                      .format(meta_idx + 1,
                              running_loss / self.args.log_iter,
                              running_acc / (self.args.log_iter * self.args.batch_size) * 100,
                              val_loss, val_acc * 100))

                running_loss = 0.
                running_acc = 0.

            # save
            if (meta_idx + 1) % self.args.save_iter == 0:
                val_loss, val_acc = self.evaluate(self.val_dataloader, self.args.num_val_tasks)
                check_pointer.update(val_acc)
                print('Checkpoint {0:d}: '
                      'val loss = {1:.4f}, val acc = {2:.2f}%'
                      .format(check_pointer.counter, val_loss, val_acc * 100))

    def test(self) -> None:
        print('Testing starts ...')
        loss_mean, loss_std, acc_mean, acc_std = self.evaluate(self.test_dataloader,
                                                               self.args.num_tst_tasks,
                                                               return_std=True)
        print('Test: nll = {0:.4f} +/- {1:.4f}, '
              'acc = {2:.2f}% +/- {3:.2f}%'
              .format(loss_mean, 1.96 * loss_std / np.sqrt(self.args.num_tst_tasks),
                      acc_mean * 100, 196 * acc_std / np.sqrt(self.args.num_tst_tasks)))

    def evaluate(self, dataloader: BatchMetaDataLoader, num_tasks: int,
                 return_std: bool = False) -> Sequence[np.array]:
        for module in self._meta_model.values():
            module.eval()   # this has no effect on the base model
        loss_list = list()
        acc_list = list()

        for eval_idx, tasks in enumerate(dataloader):
            if eval_idx >= num_tasks:
                break

            for trn_input, trn_target, tst_input, tst_target in zip(*split_tasks(tasks, self.args.device)):
                params = self.adapt(trn_input, trn_target, first_order=True)
                with torch.no_grad():
                    tst_logit = self._base_model(tst_input, params=params)
                    loss_list.append(self._nll(tst_logit, tst_target).item())
                    acc_list.append((tst_logit.argmax(dim=1) == tst_target).float().mean().item())

        if return_std:
            return np.mean(loss_list), np.std(loss_list), np.mean(acc_list), np.std(acc_list)
        else:
            return np.mean(loss_list), np.mean(acc_list)
