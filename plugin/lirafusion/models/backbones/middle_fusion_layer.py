"""
Used for fuse the middle features of lidar and radar weightedly
"""
import torch
from torch import nn
from torch.nn import functional as F

# Initialize a Registry
from mmcv.utils import Registry
MIDDLE_FUSION_LAYERS =  Registry('middle_fusion_layer')

# Create a build function from config file
def build_middle_fusion_layer(cfg):
    """Build backbone."""
    return MIDDLE_FUSION_LAYERS.build(cfg)


@MIDDLE_FUSION_LAYERS.register_module()
class Lidar_Radar_middle_fusion(nn.Module):
    """
    Fuse feature maps from lidar and radar
    """
    def __init__(self, use_Radar, use_Lidar, in_channels_1, H_1, W_1, in_channels_2, H_2, W_2):
        super(Lidar_Radar_middle_fusion, self).__init__()
        self.use_Radar = use_Radar
        self.use_Lidar = use_Lidar
        self.in_channels_1 = int(in_channels_1)
        self.H_1 = int(H_1)
        self.W_1 = int(W_1)
        self.in_channels_2 = int(in_channels_2)
        self.H_2 = int(H_2)
        self.W_2 = int(W_2)
        """
        Directly concatenation
        """
        # pass

        """
        Adaptive Gated Fusion Network
        Backbone Fusion
        "3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection"
        https://github.com/rasd3/3D-CVF
        "Robust Deep Multi-modal Learning Based on Gated Information Fusion Network"
        https://arxiv.org/pdf/1807.06233.pdf
        """
        """
        View-wise, weight shape (B, 1, H, W)
        """
        # if self.use_Lidar and self.use_Radar:
        #     # Scale 1: (B, 128, 128, 128)
        #     self.input_conv_lidar_scale_1 = nn.Conv2d(2*self.in_channels_1, 1, kernel_size=3, stride=1, padding=1)
        #     self.sigmoid_lidar_scale_1 = nn.Sigmoid()
        #     self.input_conv_radar_scale_1 = nn.Conv2d(2*self.in_channels_1, 1, kernel_size=3, stride=1, padding=1)
        #     self.sigmoid_radar_scale_1 = nn.Sigmoid()

        #     # Scale 2: (B, 256, 64, 64)
        #     self.input_conv_lidar_scale_2 = nn.Conv2d(2*self.in_channels_2, 1, kernel_size=3, stride=1, padding=1)
        #     self.sigmoid_lidar_scale_2 = nn.Sigmoid()
        #     self.input_conv_radar_scale_2 = nn.Conv2d(2*self.in_channels_2, 1, kernel_size=3, stride=1, padding=1)
        #     self.sigmoid_radar_scale_2 = nn.Sigmoid()
        
        """
        Channel-wise, weight shape (B, C, H, W)
        """
        if self.use_Lidar and self.use_Radar:
            # Scale 1: (B, 128, 128, 128)
            self.input_conv_lidar_scale_1 = nn.Conv2d(2*self.in_channels_1, self.in_channels_1, kernel_size=3, stride=1, padding=1)
            self.sigmoid_lidar_scale_1 = nn.Sigmoid()
            self.input_conv_radar_scale_1 = nn.Conv2d(2*self.in_channels_1, self.in_channels_1, kernel_size=3, stride=1, padding=1)
            self.sigmoid_radar_scale_1 = nn.Sigmoid()

            # Scale 2: (B, 256, 64, 64)
            self.input_conv_lidar_scale_2 = nn.Conv2d(2*self.in_channels_2, self.in_channels_2, kernel_size=3, stride=1, padding=1)
            self.sigmoid_lidar_scale_2 = nn.Sigmoid()
            self.input_conv_radar_scale_2 = nn.Conv2d(2*self.in_channels_2, self.in_channels_2, kernel_size=3, stride=1, padding=1)
            self.sigmoid_radar_scale_2 = nn.Sigmoid()


    def forward(self, middle_feat_lidar, middle_feat_radar):
        # Radar only
        if (middle_feat_lidar is None) or (self.use_Lidar is False):
            return middle_feat_radar
        
        # Lidar only
        if (middle_feat_radar is None) or (self.use_Radar is False):
            return middle_feat_lidar
        
        """
        Directly concatenation
        """
        # middle_feat_fused_1 = torch.cat((middle_feat_lidar[0], middle_feat_radar[0]), dim=1) # (B, 256, 128, 128)
        # middle_feat_fused_2 = torch.cat((middle_feat_lidar[1], middle_feat_radar[1]), dim=1) # (B, 512, 64, 64)
        # middle_feat_fused = [middle_feat_fused_1, middle_feat_fused_2]


        """
        Adaptive Gated Fusion Network
        "3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection"
        https://github.com/rasd3/3D-CVF
        "Robust Deep Multi-modal Learning Based on Gated Information Fusion Network"
        https://arxiv.org/pdf/1807.06233.pdf
        """
        """
        View-wise, weight shape (B, 1, H, W)
        """
        # # Scale 1: (B, 128, 128, 128)
        # # Learning the weighted generation network
        # input_feat_cat_1 = torch.cat((middle_feat_lidar[0], middle_feat_radar[0]), dim=1) # (B, 256, 128, 128)
        # weight_lidar_1 = self.input_conv_lidar_scale_1(input_feat_cat_1) 
        # weight_lidar_1 = self.sigmoid_lidar_scale_1(weight_lidar_1).repeat(1, self.in_channels_1, 1, 1) # (B, 128, 128, 128)
        # weight_radar_1 = self.input_conv_radar_scale_1(input_feat_cat_1)
        # weight_radar_1 = self.sigmoid_radar_scale_1(weight_radar_1).repeat(1, self.in_channels_1, 1, 1) # (B, 128, 128, 128)

        # # Element-wise product
        # product_lidar_feat_1 = middle_feat_lidar[0] * weight_lidar_1 # (B, 128, 128, 128)
        # product_radar_feat_1 = middle_feat_radar[0] * weight_radar_1 # (B, 128, 128, 128)

        # # Concatenation and Convolution
        # middle_feat_fused_1 = torch.cat((product_lidar_feat_1, product_radar_feat_1), dim=1) # (B, 256, 128, 128)
        # # middle_feat_fused_1 = self.output_conv_scale_1(middle_feat_fused_1) # (B, 256, 128, 128)
        # # middle_feat_fused_1 = self.output_relu_scale_1(middle_feat_fused_1)


        # # Scale 2: (B, 256, 64, 64)
        # # Learning the weighted generation network
        # input_feat_cat_2 = torch.cat((middle_feat_lidar[1], middle_feat_radar[1]), dim=1) # (B, 512, 64, 64)
        # weight_lidar_2 = self.input_conv_lidar_scale_2(input_feat_cat_2) 
        # weight_lidar_2 = self.sigmoid_lidar_scale_2(weight_lidar_2).repeat(1, self.in_channels_2, 1, 1) # (B, 256, 64, 64)
        # weight_radar_2 = self.input_conv_radar_scale_2(input_feat_cat_2)
        # weight_radar_2 = self.sigmoid_radar_scale_2(weight_radar_2).repeat(1, self.in_channels_2, 1, 1) # (B, 256, 64, 64)

        # # Element-wise product
        # product_lidar_feat_2 = middle_feat_lidar[1] * weight_lidar_2 # (B, 256, 64, 64)
        # product_radar_feat_2 = middle_feat_radar[1] * weight_radar_2 # (B, 256, 64, 64)

        # # Concatenation and Convolution
        # middle_feat_fused_2 = torch.cat((product_lidar_feat_2, product_radar_feat_2), dim=1) # (B, 512, 64, 64)
        # # middle_feat_fused_2 = self.output_conv_scale_2(middle_feat_fused_2) # (B, 512, 64, 64)
        # # middle_feat_fused_2 = self.output_relu_scale_2(middle_feat_fused_2)


        # # Final output list
        # middle_feat_fused = [middle_feat_fused_1, middle_feat_fused_2]

        """
        Channel-wise, weight shape (B, C, H, W)
        """
        # Scale 1: (B, 128, 128, 128)
        # Learning the weighted generation network
        input_feat_cat_1 = torch.cat((middle_feat_lidar[0], middle_feat_radar[0]), dim=1) # (B, 256, 128, 128)
        weight_lidar_1 = self.input_conv_lidar_scale_1(input_feat_cat_1) 
        weight_lidar_1 = self.sigmoid_lidar_scale_1(weight_lidar_1) # (B, 128, 128, 128)
        weight_radar_1 = self.input_conv_radar_scale_1(input_feat_cat_1)
        weight_radar_1 = self.sigmoid_radar_scale_1(weight_radar_1) # (B, 128, 128, 128)

        # Element-wise product
        product_lidar_feat_1 = middle_feat_lidar[0] * weight_lidar_1 # (B, 128, 128, 128)
        product_radar_feat_1 = middle_feat_radar[0] * weight_radar_1 # (B, 128, 128, 128)

        # Concatenation and Convolution
        middle_feat_fused_1 = torch.cat((product_lidar_feat_1, product_radar_feat_1), dim=1) # (B, 256, 128, 128)
        # middle_feat_fused_1 = self.output_conv_scale_1(middle_feat_fused_1) # (B, 256, 128, 128)
        # middle_feat_fused_1 = self.output_relu_scale_1(middle_feat_fused_1)


        # Scale 2: (B, 256, 64, 64)
        # Learning the weighted generation network
        input_feat_cat_2 = torch.cat((middle_feat_lidar[1], middle_feat_radar[1]), dim=1) # (B, 512, 64, 64)
        weight_lidar_2 = self.input_conv_lidar_scale_2(input_feat_cat_2) 
        weight_lidar_2 = self.sigmoid_lidar_scale_2(weight_lidar_2) # (B, 256, 64, 64)
        weight_radar_2 = self.input_conv_radar_scale_2(input_feat_cat_2)
        weight_radar_2 = self.sigmoid_radar_scale_2(weight_radar_2) # (B, 256, 64, 64)

        # Element-wise product
        product_lidar_feat_2 = middle_feat_lidar[1] * weight_lidar_2 # (B, 256, 64, 64)
        product_radar_feat_2 = middle_feat_radar[1] * weight_radar_2 # (B, 256, 64, 64)

        # Concatenation and Convolution
        middle_feat_fused_2 = torch.cat((product_lidar_feat_2, product_radar_feat_2), dim=1) # (B, 512, 64, 64)
        # middle_feat_fused_2 = self.output_conv_scale_2(middle_feat_fused_2) # (B, 512, 64, 64)
        # middle_feat_fused_2 = self.output_relu_scale_2(middle_feat_fused_2)


        # Final output list
        middle_feat_fused = [middle_feat_fused_1, middle_feat_fused_2]


        return middle_feat_fused


