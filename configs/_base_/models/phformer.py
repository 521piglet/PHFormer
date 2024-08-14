
from yacs.config import CfgNode as CN

_C = CN()
_C.name = 'phformer'
_C.rot_type = 'quat'
_C.pc_feat_dim = 128

_C.encoder = 'pointnet2_ssg'  # 'dgcnn', 'pointnet2_ssg', 'pointnet2_msg'

_C.transformer_pos_enc = (128, 128)
_C.transformer_feat_dim = 128
_C.transformer_heads = 8
_C.transformer_layers = 2
_C.transformer_pre_ln = True

_C.pose_pc_feat = False  # pose regressor input part points feature

_C.refine_steps = 1


def get_cfg_defaults():
    return _C.clone()
