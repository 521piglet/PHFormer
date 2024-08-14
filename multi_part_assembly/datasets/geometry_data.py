import os
from glob import glob
import random
import copy

import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree

from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import pdb
from tqdm import tqdm

class GeometryPartDataset(Dataset):
    """Geometry part assembly dataset.
    We follow the data prepared by Breaking Bad dataset:
        https://breaking-bad-dataset.github.io/
    """

    def __init__(
        self,
        data_dir,
        data_fn,
        data_keys,
        category='',
        num_points=1000,
        min_num_part=2,
        max_num_part=20,
        shuffle_parts=False,
        rot_range=-1,
        overfit=-1,
        pre_compute=False
    ):
        # store parameters
        self.category = category if category.lower() != 'all' else ''
        self.data_dir = data_dir
        self.data_fn = data_fn
        self.num_points = num_points
        self.min_num_part = min_num_part
        self.max_num_part = max_num_part  # ignore shapes with more parts
        self.shuffle_parts = shuffle_parts  # shuffle part orders
        self.rot_range = rot_range  # rotation range in degree

        # list of fracture folder path
        self.data_list = self._read_data(data_fn)
        if overfit > 0:
            self.data_list = self.data_list[:overfit]

        # additional data to load, e.g. ('part_ids', 'instance_label')
        self.data_keys = data_keys
        if pre_compute:
            self._get_adjacent_labels() # compute adjacent labels
            self._save_meshes() # save downsampled point clouds as npy file to accelerate reading speed
    def _get_adjacent_labels(self):
        for index in tqdm(range(len(self.data_list))):
            data_folder = os.path.join(self.data_dir, self.data_list[index])
            all_mesh_files = os.listdir(data_folder)
            mesh_files = []
            for file in all_mesh_files:
                if '.obj' in file:
                    mesh_files.append(file)
            mesh_files.sort()
            if not self.min_num_part <= len(mesh_files) <= self.max_num_part:
                raise ValueError
            # read mesh and sample points

            meshes = [
                o3d.io.read_triangle_mesh(os.path.join(data_folder, mesh_file))
                for mesh_file in mesh_files
            ]
            n = len(meshes)
            labels = np.zeros([n, n])
            for i in range(n - 1):
                for j in range(i + 1, n):
                    src_pc = o3d.geometry.PointCloud(meshes[i].vertices)
                    tgt_pc = o3d.geometry.PointCloud(meshes[j].vertices)
                    d = src_pc.compute_point_cloud_distance(tgt_pc)
                    count = (np.asarray(d) == 0).sum()
                    labels[i, j] = count
                    labels[j, i] = count
            max_id = labels.argmax(-1)
            adjacent_label = labels > 10  # 10
            x = np.arange(0, n)
            adjacent_label[x, max_id] = 1
            adjacent_label = (adjacent_label + adjacent_label.T) > 0
            np.save(os.path.join(data_folder, 'adjacent_labels.npy'), adjacent_label)

    def _save_meshes(self):
        from tqdm import tqdm
        for index in tqdm(range(len(self.data_list))):
            data_folder = os.path.join(self.data_dir, self.data_list[index])
            pcs_file_name = os.path.join(data_folder, 'pcs.npy')
            exist_flag = os.path.exists(pcs_file_name)
            if exist_flag:
                continue
            mesh_files = os.listdir(data_folder)
            mesh_files.sort()
            while '.npy' in mesh_files[0]:
                mesh_files.pop(0)
            meshes = [
                o3d.io.read_triangle_mesh(os.path.join(data_folder, mesh_file))
                for mesh_file in mesh_files
            ]
            pcs = [
                np.asarray(mesh.sample_points_uniformly(self.num_points).points)
                for mesh in meshes
            ]
            pcs = np.stack(pcs, axis=0)
            np.save(os.path.join(data_folder, 'pcs.npy'), pcs)

    def _read_data(self, data_fn):
        """Filter out invalid number of parts."""
        with open(os.path.join(self.data_dir, data_fn), 'r') as f:
            mesh_list = [line.strip() for line in f.readlines()]
            if self.category:
                mesh_list = [
                    line for line in mesh_list
                    if self.category in line.split('/')
                ]

        data_list = []

        for mesh in mesh_list:
            mesh_dir = os.path.join(self.data_dir, mesh)

            if not os.path.isdir(mesh_dir):
                print(f'{mesh} does not exist')
                continue
            fracs = os.listdir(mesh_dir)
            fracs.sort()
            for frac in fracs:
                # we take both fractures and modes for training
                if 'fractured' not in frac and 'mode' not in frac:
                    continue
                frac = os.path.join(mesh, frac)
                num_parts = len(glob(os.path.join(self.data_dir, frac, '*.obj')))
                if self.min_num_part <= num_parts <= self.max_num_part:
                    data_list.append(frac)
        return data_list

    @staticmethod
    def _recenter_pc(pc):
        """pc: [N, 3]"""
        centroid = np.mean(pc, axis=0)
        rand_t = np.random.rand(3) * 0.05
        centroid = centroid + rand_t
        pc = pc - centroid[None]

        return pc, centroid

    def _rotate_pc(self, pc):
        """pc: [N, 3]"""
        if self.rot_range > 0.:
            rot_euler = (np.random.rand(3) - 0.5) * 2. * self.rot_range
            rot_mat = R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
        else:
            rot_mat = R.random().as_matrix()
        pc = (rot_mat @ pc.T).T
        quat_gt = R.from_matrix(rot_mat.T).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[[3, 0, 1, 2]]
        return pc, quat_gt, rot_mat.T

    @staticmethod
    def _shuffle_pc(pc):
        """pc: [N, 3]"""
        order = np.arange(pc.shape[0])
        random.shuffle(order)
        pc = pc[order]
        return pc

    def _pad_data(self, data):
        """Pad data to shape [`self.max_num_part`, data.shape[1], ...]."""
        data = np.array(data)
        pad_shape = (self.max_num_part, ) + tuple(data.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[:data.shape[0]] = data
        return pad_data


    def _get_pcs(self, data_folder):
        """Read mesh and sample point cloud from a folder."""
        # `data_folder`: xxx/plate/1d4093ad2dfad9df24be2e4f911ee4af/fractured_0
        data_folder = os.path.join(self.data_dir, data_folder)

        adj_file_name = os.path.join(data_folder, 'adjacent_labels.npy')

        pcs_file = os.path.join(data_folder, 'pcs.npy')
        adjacent_label = np.load(adj_file_name)
        pcs = np.load(pcs_file)

        order = np.arange(len(pcs))
        if self.shuffle_parts:
            random.shuffle(order)
        pcs = pcs[order]

        adjacent_label = adjacent_label[order, :]
        adjacent_label = adjacent_label[:, order]
        pad_adj_label = np.zeros([self.max_num_part, self.max_num_part])
        pad_adj_label[:adjacent_label.shape[0], :adjacent_label.shape[1]] = adjacent_label

        return pcs, pad_adj_label.astype(np.float32), None

    def cal_rel_pose(self, cur_mat, cur_trans, adj_matrix):
        valid_n = len(cur_mat)
        poses = []

        for i in range(valid_n):
            pose = np.eye(4)
            pose[:3, :3] = cur_mat[i]
            pose[:3, 3] = cur_trans[i]
            poses.append(pose)
        poses = np.stack(poses, 0)
        src_idx, tgt_idx = np.where(adj_matrix)
        assert src_idx.shape[0] > 0
        src_pose, tgt_pose = poses[src_idx], poses[tgt_idx]

        rel_pose = np.linalg.inv(tgt_pose) @ src_pose
        rel_rot = rel_pose[:, :3, :3]
        rel_trans = rel_pose[:, :3, 3]

        quat_gt = R.from_matrix(rel_rot).as_quat()
        # we use scalar-first quaternion
        quat_gt = quat_gt[:, [3, 0, 1, 2]]

        pad_rel_pose = np.zeros([adj_matrix.shape[0], adj_matrix.shape[0], 7])

        pad_rel_pose[src_idx, tgt_idx] = np.concatenate([quat_gt, rel_trans], -1)
        return pad_rel_pose.astype(np.float32)


    def __getitem__(self, index):
        pcs, adj_label, merge_pc = self._get_pcs(self.data_list[index])

        num_parts = pcs.shape[0]
        cur_pts, cur_quat, cur_trans, cur_mat = [], [], [], []

        for i in range(num_parts):
            pc = pcs[i]
            pc, gt_trans = self._recenter_pc(pc)
            pc, gt_quat, gt_mat = self._rotate_pc(pc)
            cur_pts.append(pc)
            cur_quat.append(gt_quat)
            cur_trans.append(gt_trans)
            cur_mat.append(gt_mat)

        gt_rel_pose = self.cal_rel_pose(cur_mat, cur_trans, adj_label)
        cur_pts = self._pad_data(np.stack(cur_pts, axis=0))  # [P, N, 3]
        cur_quat = self._pad_data(np.stack(cur_quat, axis=0))  # [P, 4]
        cur_trans = self._pad_data(np.stack(cur_trans, axis=0))  # [P, 3]

        """
        data_dict = {
            'part_pcs': MAX_NUM x N x 3
                The points sampled from each part.

            'part_trans': MAX_NUM x 3
                Translation vector

            'part_quat': MAX_NUM x 4
                Rotation as quaternion.

            'part_valids': MAX_NUM
                1 for shape parts, 0 for padded zeros.

            'instance_label': MAX_NUM x 0, useless

            'part_label': MAX_NUM x 0, useless

            'part_ids': MAX_NUM, useless

            'data_id': int
                ID of the data.

        }
        """
        data_dict = {
            'part_pcs': cur_pts, # P, N, 3
            'part_quat': cur_quat, # P, 4
            'part_trans': cur_trans, # P, 3
            'adj_labels': adj_label, # P, P,
            'gt_rel_pose': gt_rel_pose, # P * P, 7

        }

        # valid part masks
        valids = np.zeros((self.max_num_part), dtype=np.float32)
        valids[:num_parts] = 1.
        data_dict['part_valids'] = valids  # P
        # data_id
        data_dict['data_id'] = index  # 1
        # instance_label is useless in non-semantic assembly
        # keep here for compatibility with semantic assembly
        # make its last dim 0 so that we concat nothing
        instance_label = np.zeros((self.max_num_part, 0), dtype=np.float32)
        data_dict['instance_label'] = instance_label # P, 0
        # # the same goes to part_label
        part_label = np.zeros((self.max_num_part, 0), dtype=np.float32)
        data_dict['part_label'] = part_label  # P, 0


        return data_dict

    def __len__(self):
        return len(self.data_list)


def build_geometry_dataloader(cfg):
    data_dict = dict(
        data_dir=cfg.data.data_dir,
        data_fn=cfg.data.data_fn.format('train'),
        data_keys=cfg.data.data_keys,
        category=cfg.data.category,
        num_points=cfg.data.num_pc_points,
        min_num_part=cfg.data.min_num_part,
        max_num_part=cfg.data.max_num_part,
        shuffle_parts=cfg.data.shuffle_parts,
        rot_range=cfg.data.rot_range,
        overfit=cfg.data.overfit,
        pre_compute=cfg.data.pre_compute
    )
    train_set = GeometryPartDataset(**data_dict)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.exp.batch_size,
        shuffle=True,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(cfg.exp.num_workers > 0),
    )

    data_dict['data_fn'] = cfg.data.data_fn.format('val')
    data_dict['shuffle_parts'] = False
    val_set = GeometryPartDataset(**data_dict)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg.exp.batch_size, # *2
        shuffle=False,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(cfg.exp.num_workers > 0),
    )
    return train_loader, val_loader
