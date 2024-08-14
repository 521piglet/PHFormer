import torch
import torch.nn as nn
import torch.nn.functional as F

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
def normalize_rot6d(rot):
    """Adopted from PyTorch3D.

    Args:
        rot: [..., 6] or [..., 2, 3]

    Returns:
        same shape where the first two 3-dim are normalized and orthogonal
    """
    if rot.shape[-1] == 3:
        unflatten = True
        rot = rot.flatten(-2, -1)
    else:
        unflatten = False
    a1, a2 = rot[..., :3], rot[..., 3:]
    b1 = F.normalize(a1, p=2, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, p=2, dim=-1)
    rot = torch.cat([b1, b2], dim=-1)  # back to [..., 6]
    if unflatten:
        rot = rot.unflatten(-1, (2, 3))
    return rot


class PoseRegressor(nn.Module):
    """MLP-based regressor for translation and rotation prediction."""

    def __init__(self, feat_dim, rot_type='quat', norm_rot=True):
        super().__init__()

        if rot_type == 'quat':
            rot_dim = 4
        elif rot_type == 'rmat':
            rot_dim = 6  # 6D representation from the CVPR'19 paper
        else:
            raise NotImplementedError(f'rotation {rot_type} is not supported')
        self.rot_type = rot_type
        self.norm_rot = norm_rot

        # self.fc_layers = nn.Sequential(
        #     nn.Linear(feat_dim, 256),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(0.2),
        # )
        self.fc_layers = MLP([feat_dim, 256, 128])
        # Rotation prediction head
        self.rot_head = nn.Linear(128, rot_dim)

        # Translation prediction head
        self.trans_head = nn.Linear(128, 3)

    def forward(self, x):
        """x: [B, C] or [B, P, C]"""
        f = self.fc_layers(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        rot = self.rot_head(f)  # [B, 4/6] or [B, P, 4/6]
        if self.norm_rot:
            if self.rot_type == 'quat':
                rot = F.normalize(rot, p=2, dim=-1)
            elif self.rot_type == 'rmat':
                rot = normalize_rot6d(rot)
        trans = self.trans_head(f)  # [B, 3] or [B, P, 3]
        return rot, trans


class StocasticPoseRegressor(PoseRegressor):
    """Stochastic pose regressor with noise injection."""

    def __init__(self, feat_dim, noise_dim, rot_type='quat', norm_rot=True):
        super().__init__(feat_dim + noise_dim, rot_type, norm_rot)

        self.noise_dim = noise_dim

    def forward(self, x):
        """x: [B, C] or [B, P, C]"""
        noise_shape = list(x.shape[:-1]) + [self.noise_dim]
        noise = torch.randn(noise_shape).type_as(x)
        x = torch.cat([x, noise], dim=-1)
        return super().forward(x)


class DualPoseRegressor(nn.Module):
    """MLP-based regressor for translation and rotation prediction."""

    def __init__(self, feat_dim, rot_type='quat', norm_rot=True):
        super().__init__()

        if rot_type == 'quat':
            rot_dim = 4
        elif rot_type == 'rmat':
            rot_dim = 6  # 6D representation from the CVPR'19 paper
        else:
            raise NotImplementedError(f'rotation {rot_type} is not supported')
        self.rot_type = rot_type
        self.norm_rot = norm_rot
        #
        self.rot_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )
        self.trans_mlp = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )
        # self.rot_mlp = MLP([feat_dim, 256, 128, 64, rot_dim], False)
        # self.trans_mlp = MLP([feat_dim, 256, 128, 64, 3], False)

        # Rotation prediction head
        self.rot_head = nn.Linear(128, rot_dim)
        #
        # # Translation prediction head
        self.trans_head = nn.Linear(128, 3)

    def forward(self, x, x1=None):
        """x: [B, C] or [B, P, C]"""
        if x1 == None:
            x1 = x
        rot_feats = self.rot_mlp(x)
        rot = self.rot_head(rot_feats)
        if self.norm_rot:
            if self.rot_type == 'quat':
                rot = F.normalize(rot, p=2, dim=-1)
            elif self.rot_type == 'rmat':
                rot = normalize_rot6d(rot)
        trans_feats = self.trans_mlp(x1) # [B, 3] or [B, P, 3]
        trans = self.trans_head(trans_feats)
        return rot, trans, rot_feats, trans_feats


class DualStocasticPoseRegressor(DualPoseRegressor):
    """Stochastic pose regressor with noise injection."""

    def __init__(self, feat_dim, noise_dim, rot_type='quat', norm_rot=True):
        super().__init__(feat_dim + noise_dim, rot_type, norm_rot)

        self.noise_dim = noise_dim

    def forward(self, x):
        """x: [B, C] or [B, P, C]"""
        noise_shape = list(x.shape[:-1]) + [self.noise_dim]
        noise = torch.randn(noise_shape).type_as(x)
        x = torch.cat([x, noise], dim=-1)
        return super().forward(x)
