import os
import pdb
import sys
import copy
import argparse
import importlib

import trimesh
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import glob
from multi_part_assembly.datasets import build_dataloader
from multi_part_assembly.models import build_model
from multi_part_assembly.utils import trans_rmat_to_pmat, trans_quat_to_pmat, \
    quaternion_to_rmat, save_pc
import open3d as o3d
from multi_part_assembly.utils import Rotation3D
@torch.no_grad()
def visualize(cfg):
    # Initialize model
    model = build_model(cfg)
    ckp = torch.load(cfg.exp.weight_file, map_location='cpu')
    model.load_state_dict(ckp['state_dict'])
    model = model.cuda().eval()
    # Initialize dataloaders
    _, val_loader = build_dataloader(cfg)
    val_dst = val_loader.dataset

    # save some predictions for visualization
    vis_lst, loss_lst = [], []
    for batch in tqdm(val_loader):
        batch = {k: v.float().cuda() for k, v in batch.items()}
        out_dict = model(batch)  # trans/rot: [B, P, 3/4/(3, 3)]
        part_quat =batch.pop('part_quat')
        batch['part_rot'] = \
            Rotation3D(part_quat, rot_type='quat')
        batch['part_pcs'] = batch['part_pcs'][..., :3]

        loss_dict, _ = model._calc_loss(out_dict, batch)  # loss is [B]
        # the criterion to cherry-pick examples
        loss = loss_dict['transform_pt_cd_loss']
        # convert all the rotations to quaternion for simplicity
        out_dict = {
            'data_id': batch['data_id'].long(),
            'pred_trans': out_dict['trans'],
            'pred_quat': out_dict['rot'].to_quat(),
            'gt_trans': batch['part_trans'],
            'gt_quat': batch['part_rot'].to_quat(),
            'part_valids': batch['part_valids'].long(),
            'pcs': batch['part_pcs'],
            'corr_matrix': out_dict['corr_matrix'],
            'adj_labels': batch['adj_labels'],
            'att_scores': out_dict['att_scores'],
            'super_xyz':out_dict['super_xyz'],
            'pred_rel_rot': out_dict['rel_rot'].to_rmat(),
            'pred_rel_trans': out_dict['rel_trans'],
        }
        out_dict = {k: v.cpu().numpy() for k, v in out_dict.items()}
        out_dict_lst = [{k: v[i]
                         for k, v in out_dict.items()}
                        for i in range(loss.shape[0])]
        vis_lst += out_dict_lst
        loss_lst.append(loss.cpu().numpy())
    loss_lst = np.concatenate(loss_lst, axis=0)

    top_idx = np.argsort(loss_lst)[:args.vis]
    # apply the predicted transforms to the original meshes and save them
    save_dir = os.path.join('vis', args.category)
    for rank, idx in enumerate(top_idx):
        out_dict = vis_lst[idx]
        data_id = out_dict['data_id']
        mesh_dir = os.path.join(val_dst.data_dir, val_dst.data_list[data_id])
        mesh_files = os.listdir(mesh_dir)
        mesh_files.sort()
        while '.np' in mesh_files[0]:
            mesh_files.pop(0)
        assert len(mesh_files) == out_dict['part_valids'].sum()
        subfolder_name = f"data_id{data_id}-{len(mesh_files)}pcs-{mesh_dir.split('/')[-1]}"
        cur_save_dir = os.path.join(save_dir, mesh_dir.split('/')[-2], subfolder_name)
        os.makedirs(cur_save_dir, exist_ok=True)

        np.save(os.path.join(cur_save_dir, 'corr_matrix'), out_dict['corr_matrix'])
        np.save(os.path.join(cur_save_dir, 'adj_labels'), out_dict['adj_labels'])
        np.save(os.path.join(cur_save_dir, 'pred_rel_trans'), out_dict['pred_rel_trans'].reshape(20, 20, 3)[:len(mesh_files),:len(mesh_files)])

        rel_rots = out_dict['pred_rel_rot']
        np.save(os.path.join(cur_save_dir, 'pred_rel_rot'), rel_rots.reshape(20, 20, 3, 3)[:len(mesh_files), :len(mesh_files)])
        for i, mesh_file in enumerate(mesh_files):
            mesh = trimesh.load(os.path.join(mesh_dir, mesh_file))
            mesh.export(os.path.join(cur_save_dir, mesh_file))
            # R^T (mesh - T) --> init_mesh
            gt_trans, gt_quat = \
                out_dict['gt_trans'][i], out_dict['gt_quat'][i]
            gt_rmat = quaternion_to_rmat(gt_quat)
            init_trans = -(gt_rmat.T @ gt_trans)
            init_rmat = gt_rmat.T
            init_pmat = trans_rmat_to_pmat(init_trans, init_rmat)
            np.save(os.path.join(cur_save_dir, f'init_T_{i}'), init_pmat)


            init_mesh = mesh.apply_transform(init_pmat)
            init_mesh.export(os.path.join(cur_save_dir, f'input_{mesh_file}'))
            init_pc = trimesh.sample.sample_surface(init_mesh,
                                                    val_dst.num_points)[0]
            save_pc(init_pc,
                    os.path.join(cur_save_dir, f'input_{mesh_file[:-4]}.ply'))
            # predicted pose
            pred_trans, pred_quat = \
                out_dict['pred_trans'][i], out_dict['pred_quat'][i]
            pred_pmat = trans_quat_to_pmat(pred_trans, pred_quat)
            pred_mesh = init_mesh.apply_transform(pred_pmat)
            pred_mesh.export(os.path.join(cur_save_dir, f'pred_{mesh_file}'))
            # pred_pc = trimesh.sample.sample_surface(pred_mesh,
            #                                         val_dst.num_points)[0]
            pc = out_dict['pcs'][i]
            pred_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
            pred_pc.transform(pred_pmat)
            o3d.io.write_point_cloud(os.path.join(cur_save_dir, f'pred_{mesh_file[:-4]}.ply'), pred_pc)
    print(f'Saving {len(top_idx)} predictions for visualization...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization script')
    parser.add_argument('--cfg_file', default=None, type=str, help='.py')
    parser.add_argument('--category', type=str, default='', help='data subset')
    parser.add_argument('--min_num_part', type=int, default=3)
    parser.add_argument('--max_num_part', type=int, default=-1)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--vis', type=int, default=20, help='visualization')
    args = parser.parse_args()

    sys.path.append(os.path.dirname(args.cfg_file))
    cfg = importlib.import_module(os.path.basename(args.cfg_file)[:-3])
    cfg = cfg.get_cfg_defaults()
    # cfg.exp.gpus = [2]
    if args.category:
        cfg.data.category = args.category
    if args.min_num_part > 0:
        cfg.data.min_num_part = args.min_num_part
    if args.max_num_part > 0:
        cfg.data.max_num_part = args.max_num_part
    if args.weight:
        cfg.exp.weight_file = args.weight
    else:
        assert cfg.exp.weight_file, 'Please provide weight to test'

    cfg_backup = copy.deepcopy(cfg)
    cfg.freeze()
    print(cfg)

    if not args.category:
        args.category = 'all'
    visualize(cfg)
