import os
import argparse
import random
import torch
import numpy as np

from src.maml import MAML
from src.meta_sgd import MetaSGD
from src.meta_curvature import MetaCurvature
from src.meta_mirror_descent import MetaMirrorDescent


_ALGORITHM = {'maml': MAML,
              'metasgd': MetaSGD,
              'metacurvature': MetaCurvature,
              'metamirrordescent': MetaMirrorDescent}


def main(args):
    suffix = '-' + str(args.num_way) + 'way' + str(args.num_supp) + 'shot'
    args.model_dir = os.path.join(args.model_dir, args.dataset.lower(), args.algorithm.lower() + suffix)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"   # for CUDA >= 10.2
    torch.use_deterministic_algorithms(True)

    for k, v in args.__dict__.items():
        print('%s: %s' % (k, v))

    args.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    algorithm = _ALGORITHM[args.algorithm.lower()](args)
    algorithm.train()
    algorithm.load_meta_model(args.algorithm.lower() + '_final.ct')
    algorithm.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setup variables')

    # dir
    parser.add_argument('--data-dir', type=str, default='./datasets/', help='Dataset directory')
    parser.add_argument('--model-dir', type=str, default='./models/', help='Save directory')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', help='Dataset')
    parser.add_argument('--download', type=bool, default=False, help='Whether to download the dataset')
    parser.add_argument('--num-way', type=int, default=5, help='Number of classes per task')
    parser.add_argument('--num-supp', type=int, default=1, help='Number of data per class (aka. shot) in support set')
    parser.add_argument('--num-qry', type=int, default=15, help='Number of data per class in query set')
    parser.add_argument('--num-val-tasks', type=int, default=1000, help='Number of tasks for meta-validation')
    parser.add_argument('--num-tst-tasks', type=int, default=1000, help='Number of tasks for meta-test')
    parser.add_argument('--seed', type=int, default=0, help='Seed for reproducibility')

    # algorithm
    parser.add_argument('--algorithm', type=str, default='MetaMirrorDescent', help='Few-shot learning methods')
    parser.add_argument('--cuda', type=bool, default=True, help='Whether to use cuda')

    # meta training params
    parser.add_argument('--first-order', type=bool, default=False, help='Whether to use first-order approximation')
    parser.add_argument('--meta-iter', type=int, default=60000, help='Number of epochs for meta training')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (episodes) of tasks to update meta-param')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--log-iter', type=int, default=200, help='Log iter')
    parser.add_argument('--num-log-tasks', type=int, default=100, help='Log iter')
    parser.add_argument('--save-iter', type=int, default=2000, help='Save iter')
    parser.add_argument('--meta-lr', type=float, default=1e-3, help='Learning rate for meta-updates')

    # task training params
    parser.add_argument('--base-model', type=str, default='CNN4', help='Base-learner model (CNN4/ResNet12)')
    parser.add_argument('--task-iter', type=int, default=5, help='Adaptation steps during meta-training')
    parser.add_argument('--task-lr', type=float, default=1e-2, help='Learning rate for task-updates')

    args_parsed = parser.parse_args()
    main(args_parsed)
