import torch
from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16
from torch import nn
from torch.nn import functional as F

from mmcv.utils import Registry
VOXEL_FUSION_LAYER = Registry('voxel_fusion_layer')

def build_voxel_fusion_layer(cfg):
    """Build backbone."""
    return VOXEL_FUSION_LAYER.build(cfg)


@VOXEL_FUSION_LAYER.register_module()
class voxel_fusion_layer(nn.Module):

    def __init__(self,
                in_channels, 
                out_channels, 
                ):
        super(voxel_fusion_layer, self).__init__() # TODO: add init config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radar_proj_layer = nn.Linear(in_channels, out_channels, bias=False) # training log 18, no bias
        # self.radar_proj_layer = nn.Linear(in_channels, out_channels, bias=True) # make this radar layer a learnable layer
        # init weight TODO: check how to init weight
        nn.init.normal_(self.radar_proj_layer.weight, mean=0, std=0.01)
        # nn.init.constant_(self.radar_proj_layer.bias, 0) # training log 18, no bias
        
    def forward(self, voxels, num_points, coors):
        '''
        # voxels: (M_overall, max_points, n_dim)
        # num_points: (M_overall)
        # coors: (M_overall, 4) including a flag of which sample within a batch (batch_idx)

        # out: (M_overall, n_dim)
        '''

        out = torch.zeros((voxels.shape[0], 9), dtype=torch.float32, device=voxels.device)
        out[:,:3] = voxels[:,:,:3].sum(dim=1) / num_points.view(-1,1).float() # average the xyz

        
        mask_lidar = voxels[:,:,-1] == -1
        mask_lidar_sum = mask_lidar.sum(dim=1)
        mask_lidar_sum_modified = torch.clamp(mask_lidar_sum, min=1)
        out[:,3:5] = torch.sum(voxels[:,:,3:5] * mask_lidar.unsqueeze(2), dim=1) / mask_lidar_sum_modified.view(-1,1).float() # average the intensity and dt

        mask_radar = voxels[:,:,-1] == 1
        if torch.sum(mask_radar) > 0:
            mask_radar_sum = mask_radar.sum(dim=1)
            mask_radar_sum_modified = torch.clamp(mask_radar_sum, min=1)
            out[:,5:] = self.radar_proj_layer(torch.sum(voxels[:,:,5:-1] * mask_radar.unsqueeze(2), dim=1) / mask_radar_sum_modified.view(-1,1).float())
        else:
            # print("no radar points") # if the first frame of a scene, no radar points
            # avoid error when training with b_size 1 on multi GPU
            out[:,5:] = self.radar_proj_layer(torch.zeros((out.shape[0], 4), device=voxels.device))
        # average based on the mask

        # out[:,5:][mask_radar_nonzero] = self.radar_proj_layer(voxels[:,:,5:-1][mask_radar_nonzero].sum(dim=1) / mask_radar_nonzero.view(-1,1).float()) # average the radar features
        # out[:,3:5][mask_lidar] = voxels[:,3:5][mask_lidar].sum(dim=1) / mask_lidar.sum(dim=1).view(-1,1).float() # average the intensity and dt


        # out[:,3:5] = voxels[:,3:5].sum(dim=1) / mask_lidar.sum(dim=1).view(-1,1).float() # average the intensity and dt
        # out[:,5:-1] = voxels[:,5:-1].sum(dim=1) / mask_radar.sum(dim=1).view(-1,1).float() # average the radar features
        
        return out

