import pdb
import torch
import torch.nn as nn

from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
# from multi_part_assembly.models.pn_transformer.TransformerBlock import VolumetricPositionEncoding
class VolumetricPositionEncoding(nn.Module):

    def __init__(self, feature_dim, voxel_size, vol_origin=[-1., -1., -1.], pe_type='sinusoidal'):
        super().__init__()

        self.feature_dim = feature_dim
        self.voxel_size = voxel_size
        self.vol_origin = vol_origin
        self.pe_type = pe_type
        div_term = torch.exp(torch.arange(0, self.feature_dim // 3 - 1, 2, dtype=torch.float) * (-9.2103 / (self.feature_dim // 3)))
        self.div_term = div_term.view(1, 1, -1)  # [1, 1, d//6]
    def voxelize(self, xyz):
        '''
        @param xyz: B,N,3
        @return: B,N,3
        '''
        if type ( self.vol_origin ) == list :
            self.vol_origin = torch.FloatTensor(self.vol_origin ).view(1, 1, -1).to( xyz.device )
        return (xyz - self.vol_origin) / self.voxel_size

    @staticmethod
    def embed_rotary(x, cos, sin):
        '''
        @param x: [B,N,d]
        @param cos: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @param sin: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @return:
        '''
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    @staticmethod
    def embed_pos(x, pe, pe_type='sinusoidal'):
        """ combine feature and position code
        """
        if pe_type == 'rotary':
            return VolumetricPositionEncoding.embed_rotary(x, pe[..., 0], pe[..., 1])
        elif pe_type == 'sinusoidal':
            B, K, C1 = pe.shape
            C = x.shape[-1]
            pad = torch.zeros([B, K, C-C1]).type_as(x)
            pe = torch.cat([pe, pad], -1)
            return x + pe
        else:
            raise KeyError()
    def forward(self,  XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape

        vox = self.voxelize(XYZ)
        x_position, y_position, z_position = vox[..., 0:1], vox[..., 1:2], vox[..., 2:3]                           #(-torch.log(10000.0)
        div_term = self.div_term.to(XYZ.device)


        sinx = torch.sin(x_position * div_term)  # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        if self.pe_type == 'sinusoidal':
            position_code = torch.cat([sinx, cosx, siny, cosy, sinz, cosz], dim=-1)
        elif self.pe_type == "rotary":
            # sin/cos [θ0,θ1,θ2......θd/6-1] -> sin/cos [θ0,θ0,θ1,θ1,θ2,θ2......θd/6-1,θd/6-1]
            sinx, cosx, siny, cosy, sinz, cosz = map(lambda feat: torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
                 [sinx, cosx, siny, cosy, sinz, cosz])
            sin_pos = torch.cat([sinx,siny,sinz], dim=-1)
            cos_pos = torch.cat([cosx,cosy,cosz], dim=-1)
            position_code = torch.stack([cos_pos, sin_pos], dim=-1)
        else:
            raise KeyError()
        if position_code.requires_grad:
            position_code = position_code.detach()

        return position_code
class PointNet2SSG(nn.Module):
    """PointNet++ feature extractor.

    Input point clouds [B, N, 3].
    Output global feature [B, feat_dim].
    """

    def __init__(self, feat_dim):
        super().__init__()

        self.feat_dim = feat_dim
        self.use_pe = False
        self.use_normal = False

        self.init_dim = 60 if self.use_pe else 0
        if self.use_pe:
            self.pe_encoder = VolumetricPositionEncoding(self.init_dim, voxel_size=0.01, pe_type='sinusoidal')
        self.init_dim = 1 if self.use_normal else 0

        self._build_model()
    # def _build_model(self):
    #     self.SA_modules = nn.ModuleList()
    #     self.SA_modules.append(
    #         PointnetSAModuleMSG(
    #             npoint=256,
    #             radii=[0.2, 0.25],
    #             nsamples=[50, 70],
    #             mlps=[[0, 32, 32, 64], [0, 64, 64, 128]],
    #             use_xyz=True,
    #         ))
    #
    #     input_channels = 64 + 128
    #     self.SA_modules.append(
    #         PointnetSAModule(
    #             npoint=64,
    #             radius=0.3,
    #             nsample=25,
    #             mlp=[input_channels, 128, 256],
    #             use_xyz=True,
    #         ))
    #     self.SA_modules.append(
    #             PointnetSAModule(
    #                 npoint=16,
    #                 radius=0.4,
    #                 nsample=10,
    #                 mlp=[256, 256, self.feat_dim],
    #                 use_xyz=True,
    #             ))
    def _build_model(self):  # for 1000 points
        use_xyz = True
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=50,    # 50 # 100
                mlp=[self.init_dim, 64, 128],
                use_xyz=True,
            ))
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.3,
                nsample=25,
                mlp=[128, 128, 256],
                use_xyz=use_xyz,
            ))

        self.SA_modules.append(
            PointnetSAModule(
                npoint=32,
                radius=0.4,
                nsample=10,
                mlp=[256, 512, self.feat_dim],
                use_xyz=use_xyz,
            ))
    # def _build_model(self):  # for 1000 points
    #     self.SA_modules = nn.ModuleList()
    #     self.SA_modules.append(
    #         PointnetSAModule(
    #             npoint=512,
    #             radius=0.15,
    #             nsample=30,
    #             mlp=[self.init_dim, 64, 128],
    #             use_xyz=True,
    #         ))
    #     self.SA_modules.append(
    #         PointnetSAModule(
    #             npoint=256,
    #             radius=0.25,
    #             nsample=25,
    #             mlp=[128, 128, 256],
    #             use_xyz=True,
    #         ))
    #     self.SA_modules.append(
    #         PointnetSAModule(
    #             npoint=128,
    #             radius=0.3,
    #             nsample=15,
    #             mlp=[256, 256, 256],
    #             use_xyz=True,
    #         ))
    #
    #     self.SA_modules.append(
    #         PointnetSAModule(
    #             npoint=32,
    #             radius=0.4,
    #             nsample=10,
    #             mlp=[256, 512, self.feat_dim],
    #             use_xyz=True,
    #         ))

    # def _build_model(self):
    #     self.SA_modules = nn.ModuleList()
    #     self.SA_modules.append(
    #         PointnetSAModule(
    #             npoint=768,
    #             radius=0.15,
    #             nsample=45,
    #             mlp=[0, 32, 64],
    #             use_xyz=True,
    #         ))
    #     self.SA_modules.append(
    #         PointnetSAModule(
    #             npoint=256,
    #             radius=0.2,
    #             nsample=35,
    #             mlp=[64, 128, 128],
    #             use_xyz=True,
    #         ))
    #     self.SA_modules.append(
    #         PointnetSAModule(
    #             npoint=64,
    #             radius=0.3,
    #             nsample=25,
    #             mlp=[128, 256, 256],
    #             use_xyz=True,
    #         ))
    #
    #     self.SA_modules.append(
    #         PointnetSAModule(
    #             npoint=16,
    #             radius=0.4,
    #             nsample=10,
    #             mlp=[256, 512, self.feat_dim],
    #             use_xyz=True,
    #         ))

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        if self.use_pe:
            features = self.pe_encoder(xyz).permute(0, 2, 1).contiguous()
        elif self.use_normal:
            features = pc[..., 3:].permute(0, 2, 1).contiguous()
        else:
            features = None
        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)  # [B, C, N]
        return features, xyz  # [B, C]
