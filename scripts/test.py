import os
import sys
import copy
import argparse
import importlib

import numpy as np

import torch
import pytorch_lightning as pl

from multi_part_assembly.datasets import build_dataloader
from multi_part_assembly.models import build_model
import logging

import pdb
@torch.no_grad()
def test(cfg):
    # Initialize model
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1024**2:.2f} MB")
    all_gpus = list(cfg.exp.gpus)
    trainer = pl.Trainer(
        gpus=all_gpus,
        # we should use DP because DDP will duplicate some data
        strategy='dp' if len(all_gpus) > 1 else None,
    )
    all_metrics = {
        'rot_rmse': 1.,
        'rot_mae': 1.,
        'trans_rmse': 100.,  # presented as \times 1e-2 in the table
        'trans_mae': 100.,  # presented as \times 1e-2 in the table
        'transform_pt_cd_loss': 1000.,  # presented as \times 1e-3 in the table
        'part_acc': 100.,  # presented in % in the table

    }
    if args.category == 'all':
        _, val_loader = build_dataloader(cfg)
        print(val_loader.dataset.data_list[:10])
        trainer.test(model, val_loader, ckpt_path=cfg.exp.weight_file)
        results = model.test_results
        results = {k[5:]: v.detach().cpu().numpy() for k, v in results.items()}
        for metric in all_metrics.keys():
            logging.info(metric + f': {results[metric] * all_metrics[metric]:.1f}')
        return

    # if `args.category` is 'all', we also compute per-category results
    # TODO: modify this to fit in the metrics you want to report
    all_category = cfg.data.all_category
    all_results = {metric: [] for metric in all_metrics.keys()}

    for cat in all_category:
        print(cat)
        cfg = copy.deepcopy(cfg_backup)
        cfg.data.category = cat
        _, val_loader = build_dataloader(cfg)
        trainer.test(model, val_loader, ckpt_path=cfg.exp.weight_file)
        results = model.test_results
        results = {k[5:]: v.detach().cpu().numpy() for k, v in results.items()}
        for metric in all_metrics.keys():
            all_results[metric].append(results[metric] * all_metrics[metric])
    all_results = {k: np.array(v).round(1) for k, v in all_results.items()}
    # format for latex table
    for metric, result in all_results.items():
        logging.info(f'{metric}:')
        result = result.tolist()
        result.append(np.mean(result).round(1))  # per-category mean
        result = [str(res) for res in result]
        logging.info(' '.join(result))
    logging.info('\n')
    print('Done testing...')


if __name__ == '__main__':
    configs = [
               'configs/phformer/phformer-32x1-cosine_400e-everyday.py',
               'configs/phformer/phformer-32x1-cosine_400e-artifact.py',
               'configs/phformer/phformer-32x1-cosine_400e-other.py',
               ]
    # weight_path = ''
    # weight_path = 'checkpoint/Htransformer-2-1000-edgeconv-32-d128-t_aug_0.05-2gpu-lr-0.0005-rel_rot_cos-cdloss_corrected-/models/model-epoch=399.ckpt'
    # weight_path = 'checkpoint/Htransformer-2-1000-edgeconv-32-d128-t_aug_0.05-2gpu-lr-0.0005-artifact_pretrain-/models/model-epoch=384.ckpt'
    # weight_path = 'checkpoint/Htransformer-2-1000-edgeconv-32-d128-t_aug_0.05-2gpu-lr-0.0005-everyday-2-100-/models/model-epoch=394.ckpt'


    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--cfg_file', default=configs[0], type=str, help='.py')
    parser.add_argument('--category', type=str, default='', help='data subset')
    parser.add_argument('--min_num_part', type=int, default=-1)
    parser.add_argument('--max_num_part', type=int, default=-1)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--weight', type=str, default='checkpoint/pretrain/everyday.ckpt', help='load weight')
    args = parser.parse_args()
    sys.path.append(os.path.dirname(args.cfg_file))
    cfg = importlib.import_module(os.path.basename(args.cfg_file)[:-3])
    cfg = cfg.get_cfg_defaults()
    cfg.exp.gpus = args.gpus
    if args.category:
        cfg.data.category = args.category
    if args.min_num_part > 0:
        cfg.data.min_num_part = args.min_num_part
    if args.max_num_part > 0:
        cfg.data.max_num_part = args.max_num_part
    if args.weight:
        cfg.exp.weight_file = args.weight
    elif cfg.model.name == 'identity':  # trivial identity model
        cfg.exp.weight_file = None  # no checkpoint needed
    else:
        assert cfg.exp.weight_file, 'Please provide weight to test'


    cfg_backup = copy.deepcopy(cfg)
    cfg.freeze()
    print(cfg)
    log_filename = args.weight.replace('ckpt', 'log')
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    test(cfg)
