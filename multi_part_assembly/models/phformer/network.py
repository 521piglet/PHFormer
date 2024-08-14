import pdb
import time
import torch
import torch.nn as nn
from multi_part_assembly.models import BaseModel
from multi_part_assembly.models import StocasticPoseRegressor, build_encoder
from multi_part_assembly.utils import _get_clones
from multi_part_assembly.utils import trans_l2_loss, rot_l1_loss,  rot_points_cd_loss, \
    stacked_shape_cd_loss, rot_cosine_loss, rot_points_l2_loss, chamfer_distance, corr_matrix_loss, corr_BCE_loss, \
    trans_metrics, rot_metrics
from multi_part_assembly.models.phformer.TransformerBlock import HybridTransformer

import itertools
import numpy as np
def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class GlobalEncoder(nn.Module):
    """Learnable positional encoding model."""

    def __init__(self, dims):
        super().__init__()
        self.layers = MLP(dims)

    def forward(self, x, valids):
        pos_enc = self.layers(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        valids = valids.unsqueeze(-1)
        global_enc = (pos_enc * valids).sum(1, keepdim=True) / valids.sum(1, keepdim=True)
        return global_enc

class RelativePoseEstimator(nn.Module):
    def __init__(self, input_dim, noise_dim, rot_type):
        super(RelativePoseEstimator, self).__init__()
        self.input_dim = input_dim
        enc_input_dim = 2 * input_dim
        self.encoder = MLP([enc_input_dim, input_dim, input_dim])
        self.rel_pose_predictor = StocasticPoseRegressor(
            feat_dim=input_dim,
            noise_dim=noise_dim,
            rot_type=rot_type,
        )
        self.adj_predictor = MLP([input_dim, input_dim//2, 1])
        self.merge_edge = MLP([2 * input_dim, input_dim, input_dim])

    def forward(self, node_feats, valid_mask):
        B, P, C = node_feats.shape
        pair_pc_feats = torch.cat([node_feats[:, :, None, :].repeat(1, 1, P, 1), node_feats[:, None, :, :].repeat(1, P, 1, 1)], -1)
        pair_pc_feats = pair_pc_feats.reshape(B, P * P, -1)

        edge_feats = self.encoder(pair_pc_feats.permute(0, 2, 1))# B C P*P

        adj_feats = self.adj_predictor(edge_feats)# B 1 P*P
        adj_scores, mask = self.cal_corr_matrix(adj_feats, valid_mask) # B P P

        edge_feats = edge_feats.permute(0, 2, 1) # B P*P C
        rel_rot, rel_trans = self.rel_pose_predictor(edge_feats)

        new_node_feats = self._edge2node(node_feats, edge_feats.reshape(B, P, P, C), adj_scores * mask)

        return rel_rot, rel_trans, new_node_feats, adj_scores
    def _edge2node(self, node_feats, edge_feats, relation_matrix):
        """Perform one step of message passing, get per-node messages."""
        B, P, _ = node_feats.shape
        # compute message as weighted sum over edge features
        part_message = edge_feats * relation_matrix.unsqueeze(-1)
        part_message = part_message.sum(dim=2)  # B x P x F
        norm = relation_matrix.sum(dim=-1, keepdim=True)  # B x P x 1
        normed_part_message = part_message / (norm + 1e-6)
        fuse_feats = torch.cat([node_feats, normed_part_message], -1)
        fuse_feats = self.merge_edge(fuse_feats.permute(0, 2, 1)).permute(0, 2, 1)
        return fuse_feats

    def cal_corr_matrix(self, corr_feats, valid_mask):
        scores = torch.sigmoid(corr_feats.squeeze(1))# B, P*P
        M = scores.reshape(valid_mask.shape[0], valid_mask.shape[1], -1)
        mask = valid_mask.unsqueeze(-1) * valid_mask.unsqueeze(1)
        return M, mask
class PHFormer(BaseModel):
    """PNTransformer with iterative refinement."""

    def __init__(self, cfg):
        self.refine_steps = cfg.model.refine_steps
        self.pose_pc_feat = cfg.model.pose_pc_feat

        super().__init__(cfg)
        self.encoder = self._init_encoder()
        self.HTransformer = self._init_Transformer()
        self.g_encoders = self._init_global_ecoder()
        self.pose_predictor = self._init_pose_predictor()
        self.rel_pose_predictor = self._initial_rel_pose_esimator()

    def _initial_rel_pose_esimator(self):
        rel_estimator = RelativePoseEstimator(self.pc_feat_dim,
                                              noise_dim=self.cfg.loss.noise_dim,
                                              rot_type=self.rot_type,)
        rel_estimators = _get_clones(rel_estimator, self.refine_steps)
        return rel_estimators

    def _init_encoder(self):
        """Part point cloud encoder."""
        encoder = build_encoder(
            self.cfg.model.encoder,
            feat_dim=self.pc_feat_dim,
            global_feat=True,
        )
        return encoder
    def _init_Transformer(self):
        layers = ['self', 'cross'] * 2
        return HybridTransformer(layers, self.pc_feat_dim, 4)

    def _init_global_ecoder(self):
        g_ecoder = GlobalEncoder([self.pc_feat_dim, self.pc_feat_dim*2, self.pc_feat_dim])
        g_ecoders = _get_clones(g_ecoder, self.refine_steps)
        return g_ecoders


    def _init_pose_predictor(self):
        """Final pose estimator."""
        # concat feature, instance_label and noise as input
        dim = self.pc_feat_dim + self.pose_dim
        if self.semantic:  # instance_label in semantic assembly
            dim += self.max_num_part
        if self.pose_pc_feat:
            dim += self.pc_feat_dim
        if self.use_part_label:
            dim += self.cfg.data.num_part_category
        pose_predictor = StocasticPoseRegressor(
            feat_dim=dim,
            noise_dim=self.cfg.loss.noise_dim,
            rot_type=self.rot_type,
        )
        pose_predictors = _get_clones(pose_predictor, self.refine_steps)
        return pose_predictors

    def _extract_part_feats(self, part_pcs, part_valids):
        """Extract per-part point cloud features."""
        B, P, N, _ = part_pcs.shape  # [B, P, N, 3]
        valid_mask = (part_valids == 1)
        # shared-weight encoder
        valid_pcs = part_pcs[valid_mask]  # [n, N, 3]
        super_feats, super_xyz = self.encoder(valid_pcs)  # [n, C, k]
        valid_feats, scores = self.HTransformer(super_feats, super_xyz, valid_mask)

        pc_feats = torch.zeros(B, P, self.pc_feat_dim).type_as(valid_feats)
        pc_feats[valid_mask] = valid_feats

        super_points = torch.zeros(B, P, super_xyz.shape[-2], super_xyz.shape[-1]).type_as(super_xyz)
        super_points[valid_mask] = super_xyz

        return pc_feats, super_points, scores


    def forward(self, data_dict):
        """Forward pass to predict poses for each part.

        Args:
            data_dict should contains:
                - part_pcs: [B, P, N, 3]
                - part_valids: [B, P], 1 are valid parts, 0 are padded parts
                - part_label: [B, P, NUM_PART_CATEGORY] when using as input
                    otherwise [B, P, 0] just a placeholder for compatibility
                - instance_label: [B, P, P (0 in geometric assembly)]
            may contains:
                - pc_feats: [B, P, C] (reused) or None
        """
        part_pcs, part_valids = data_dict['part_pcs'], data_dict['part_valids']
        pc_feats, super_xyz, scores = self._extract_part_feats(part_pcs, part_valids)

        part_feats = pc_feats
        part_label = data_dict['part_label'].type_as(pc_feats)
        inst_label = data_dict['instance_label'].type_as(pc_feats)
        B, P, _ = inst_label.shape
        # init pose as identity
        pose = self.zero_pose.repeat(B, P, 1).type_as(part_feats).detach()

        pred_rot, pred_trans = [], []
        pred_rel_rot, pred_rel_trans, pred_adj_scores = [], [], []
        for i in range(self.refine_steps):
            valid_mask = (part_valids == 1)
            rel_rot, rel_trans, node_feats, adj_scores = self.rel_pose_predictor[i](part_feats, valid_mask)
            pred_rel_trans.append(rel_trans)
            pred_rel_rot.append(rel_rot)
            pred_adj_scores.append(adj_scores)

            global_feats = self.g_encoders[i](node_feats, valid_mask)

            node_feats1 = node_feats + global_feats
            # MLP predict poses
            node_feats2 = torch.cat([node_feats1, part_label, inst_label, pose], dim=-1)
            if self.pose_pc_feat:
                node_feats2 = torch.cat([pc_feats, node_feats2], dim=-1)
            rot, trans = self.pose_predictor[i](node_feats2)
            pred_rot.append(rot)
            pred_trans.append(trans)

            # update for next iteration
            pose = torch.cat([rot, trans], dim=-1)
            part_feats = node_feats1

        if self.training:
            pred_rot = self._wrap_rotation(torch.stack(pred_rot, dim=0))
            pred_trans = torch.stack(pred_trans, dim=0)
            pred_rel_rot = self._wrap_rotation(torch.stack(pred_rel_rot, dim=0))
            pred_rel_trans = torch.stack(pred_rel_trans, dim=0)
            pred_adj_scores = torch.stack(pred_adj_scores, dim=0)

        else:
            # directly take the last step results
            pred_rot = self._wrap_rotation(pred_rot[-1])
            pred_trans = pred_trans[-1]
            pred_rel_rot = self._wrap_rotation(pred_rel_rot[-1])
            pred_rel_trans = pred_rel_trans[-1]
            pred_adj_scores = pred_adj_scores[-1]

        pred_dict = {
            'rot': pred_rot,  # [(T, )B, P, 4/(3, 3)], Rotation3D
            'trans': pred_trans,  # [(T, )B, P, 3]
            'rel_rot': pred_rel_rot,  # [(T, )B, P, 4/(3, 3)], Rotation3D
            'rel_trans': pred_rel_trans,  # [(T, )B, P, 3]
            'pc_feats': pc_feats,  # [B, P, C]
            'corr_matrix': pred_adj_scores,
            'att_scores': scores,
            'super_xyz': super_xyz,
        }
        return pred_dict

    def _loss_function(self, data_dict, out_dict={}, optimizer_idx=-1):
        """Predict poses and calculate loss."""
        part_pcs, valids = data_dict['part_pcs'], data_dict['part_valids']
        forward_dict = {
            'part_pcs': part_pcs,
            'part_valids': valids,
            'part_label': data_dict['part_label'],
            'instance_label': data_dict['instance_label'],
            'pc_feats': out_dict.get('pc_feats', None),
        }

        # prediction
        tic_time = time.time()
        out_dict = self.forward(forward_dict)
        toc_time = time.time()
        pc_feats = out_dict['pc_feats']
        # loss computation
        data_dict['part_pcs'] = part_pcs[..., :3]
        if not self.training:
            rel_loss_dict = self._calc_rel_loss(out_dict, data_dict)
            loss_dict, out_dict = self._calc_loss(out_dict, data_dict)
            out_dict['pc_feats'] = pc_feats
            loss_dict.update(rel_loss_dict)
            run_time = (toc_time - tic_time) / part_pcs.shape[0]
            run_time = torch.tensor([run_time], device='cuda').repeat(part_pcs.shape[0])
            loss_dict['time'] = run_time
            return loss_dict, out_dict

        pred_trans, pred_rot = out_dict['trans'], out_dict['rot']
        pred_rel_trans, pred_rel_rot, corr_matrix = out_dict['rel_trans'], out_dict['rel_rot'], out_dict['corr_matrix']
        all_loss_dict = None
        for i in range(self.refine_steps):
            pred_dict = {'rot': pred_rot[i], 'trans': pred_trans[i],
                         'rel_rot': pred_rel_rot[i], 'rel_trans': pred_rel_trans[i],
                         'corr_matrix': corr_matrix[i]}
            rel_loss_dict = self._calc_rel_loss(pred_dict, data_dict)

            loss_dict, out_dict = self._calc_loss(pred_dict, data_dict)
            loss_dict.update(rel_loss_dict)

            if all_loss_dict is None:
                all_loss_dict = {k: 0. for k in loss_dict.keys()}
            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict[k] + v
                all_loss_dict[f'{k}_{i}'] = v
        out_dict['pc_feats'] = pc_feats

        return all_loss_dict, out_dict
    def _calc_rel_loss(self, out_dict, data_dict):
        """Calculate loss by matching GT to prediction.

        Also compute evaluation metrics during testing.
        """
        pred_trans, pred_rot = out_dict['rel_trans'], out_dict['rel_rot']

        # matching GT with predictions for lowest loss in semantic assembly
        valids, part_pcs = data_dict['part_valids'], data_dict['part_pcs']
        gt_pose = data_dict['gt_rel_pose']
        corr_matrix, adj_labels = out_dict['corr_matrix'], data_dict['adj_labels']

        new_trans, new_rot = \
            gt_pose[..., 4:].detach().clone(), gt_pose[..., :4].detach().clone()

        # cal corr matrix loss using l2
        corr_loss = corr_matrix_loss(corr_matrix, adj_labels, valids)
        # cal rot cosine loss
        rel_valids = adj_labels.reshape(valids.shape[0], -1)
        gt_rot = self._wrap_rotation(new_rot)


        rot_loss = rot_cosine_loss(pred_rot, gt_rot, rel_valids)
        # cal trans l2 loss
        trans_loss = trans_l2_loss(pred_trans, new_trans, rel_valids)


        # return some intermediate variables for reusing
        loss_dict = {
            'corr_loss': corr_loss,  # [B, P, 3]
            'rel_rot_loss': rot_loss,  # [B, P, 4]
            'rel_trans_loss': trans_loss,  # [B, P, N, 3]
        }
        if not self.training:
            metric_dict = {}
            pred_corr = corr_matrix > 0.5
            pred_corr = pred_corr * valids[:, :, None] * valids[:, None, :]
            is_equal = (pred_corr != adj_labels).sum(-1).sum(-1)
            valid_n = valids.sum(-1)
            class_acc = 1 - is_equal / (valid_n**2-valid_n)
            metric_dict['class_acc'] = class_acc
            for metric in ['rmse', 'mae']:
                metric_dict[f'rel_trans_{metric}'] = trans_metrics(
                    pred_trans, new_trans, rel_valids, metric=metric)
                metric_dict[f'rel_rot_{metric}'] = rot_metrics(
                    pred_rot, gt_rot, rel_valids, metric=metric)
            loss_dict.update(metric_dict)
        return loss_dict