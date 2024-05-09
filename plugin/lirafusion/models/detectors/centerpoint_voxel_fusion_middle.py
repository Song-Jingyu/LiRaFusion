import time
import torch
import numpy as np
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from plugin.lirafusion.models.backbones.voxel_fusion_layer import build_voxel_fusion_layer

from plugin.lirafusion.models.backbones.middle_fusion_layer import build_middle_fusion_layer
# from plugin.lirafusion.models.backbones.middle_fusion_layer_neck import build_middle_fusion_layer_neck

from mmdet3d.models.builder import build_middle_encoder
from mmdet3d.models.builder import build_voxel_encoder
from mmdet3d.models.builder import build_backbone
from mmdet3d.models.builder import build_neck

from mmcv.ops import Voxelization
from mmdet.core import multi_apply
from mmcv.runner import force_fp32
from torch.nn import functional as F





@DETECTORS.register_module()
class CenterPoint_voxel_fusion_middle(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 voxel_fusion_layer=None,
                 voxel_fusion=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 middle_fusion=False,
                 pts_voxel_layer_radar=None,
                 pts_voxel_encoder_radar=None,
                 pts_middle_encoder_radar=None,
                 pts_backbone_radar=None,
                 middle_fusion_layer=None,
                 pts_neck_radar=None,
                 use_radar=False,
                 use_LiDAR=True,
                 use_Cam=False,
                 two_stage_fusion=False,
                 ):
        super(CenterPoint_voxel_fusion_middle,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)
        self.voxel_fusion = voxel_fusion
        self.voxel_fusion_layer = build_voxel_fusion_layer(voxel_fusion_layer)
        self.middle_fusion = middle_fusion
        self.use_Radar = use_radar
        self.use_LiDAR = use_LiDAR
        self.use_Cam = use_Cam
        self.two_stage_fusion = two_stage_fusion

        # early fusion
        if not self.middle_fusion:
            raise NotImplementedError
        
        # middle fusion whatever one-stage fusion or two-stage fusion!!
        else:
            if self.use_Radar:
                if pts_voxel_layer_radar:
                    # Radar voxel layer
                    self.pts_voxel_layer_radar = Voxelization(**pts_voxel_layer_radar)
                if pts_voxel_encoder_radar:
                    # Radar encoder
                    self.pts_voxel_encoder_radar = build_voxel_encoder(pts_voxel_encoder_radar)
                if pts_middle_encoder_radar:
                    # Radar middle encoder
                    self.pts_middle_encoder_radar = build_middle_encoder(pts_middle_encoder_radar)
                if pts_backbone_radar:
                    # Radar backbone
                    self.pts_backbone_radar = build_backbone(pts_backbone_radar)
                if pts_neck_radar:
                    # Radar neck
                    self.pts_neck_radar = build_neck(pts_neck_radar)
            
            # pts_backbone_radar is a flag for middle fusion after middle encoder
            if middle_fusion_layer and pts_backbone_radar and not pts_neck_radar:
                # Middle fusion layer after backbone
                self.middle_fusion_layer = build_middle_fusion_layer(middle_fusion_layer)
            
            # if middle_fusion_layer and pts_backbone_radar and pts_neck_radar:
            #     # Middle fusion layer after neck
            #     self.middle_fusion_layer = build_middle_fusion_layer_neck(middle_fusion_layer)

    """
    Voxelization function inherited from Base class
    """
    @torch.no_grad()
    @force_fp32()
    def voxelize_radar(self, points):
        """Apply dynamic voxelization to points.
        Args:
            points (list[torch.Tensor]): Points of each sample.
        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer_radar(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        
        return voxels, num_points, coors_batch
    
    @property
    def with_velocity(self):
        """bool: Whether the head predicts velocity"""
        return self.pts_bbox_head is not None and \
            self.pts_bbox_head.with_velocity

    """
    Used for middle fusion after Second(default)
    Extract Lidar and Radar features from lidar_pts_lst and radar_pts_lst
    """
    def extract_lidar_radar_feat_middle_fusion(self, lidar_pts_lst, radar_pts_lst):
        # assert (use_Lidar or use_Radar = True)!!! in extract_feat_v3 unction
        if not self.with_pts_bbox:
            return None

        # Lidar: Voxelize -----> Middle encoder
        if self.use_LiDAR:
            # Voxelize
            voxels_lidar, num_points_lidar, coors_lidar = self.voxelize(lidar_pts_lst)
            # Lidar encoder
            # voxels_feat_lidar = self.pts_voxel_encoder(voxels_lidar, num_points_lidar, coors_lidar)
            voxels_feat_lidar = self.voxel_fusion_layer(voxels_lidar, num_points_lidar, coors_lidar)
            # Middle encoder
            # middle_feat_lidar: (bs, C*D, H, W)
            batch_size_lidar = coors_lidar[-1, 0] + 1
            middle_feat_lidar = self.pts_middle_encoder(voxels_feat_lidar, coors_lidar, batch_size_lidar.item())
            backbone_feat_lidar = self.pts_backbone(middle_feat_lidar) # list
        else:
            backbone_feat_lidar = None
        
        # Radar: Voxelize ------> Middle encoder
        if self.use_Radar:
            # Voxelize
            voxels_radar, num_points_radar, coors_radar = self.voxelize_radar(radar_pts_lst)
            # Lidar encoder
            voxels_feat_radar = self.pts_voxel_encoder_radar(voxels_radar, num_points_radar, coors_radar)
            # Middle encoder
            # middle_feat_lidar: (bs, C*D, H, W)
            batch_size_radar = coors_radar[-1, 0] + 1
            middle_feat_radar = self.pts_middle_encoder_radar(voxels_feat_radar, coors_radar, batch_size_radar.item())
            bakcbone_feat_radar = self.pts_backbone_radar(middle_feat_radar) # list
        else:
            bakcbone_feat_radar = None
        
        # Fuse the BEV features of lidar and radar
        x = self.middle_fusion_layer(backbone_feat_lidar, bakcbone_feat_radar)

        # Go through backbone and fpn
        if self.with_pts_neck:
            x = self.pts_neck(x)
        
        return x
    
    def extract_feat_v3(self, lidar_pts, img, img_metas, fused_points=None, radar=None, mode='train'):
        """Extract features from images and points."""
        # Preprocess the lidar and radar data
        if self.use_LiDAR:
            if mode == 'train':
                lidar_pts_lst = fused_points
            elif mode == 'test':
                lidar_pts_lst = fused_points
        else:
            lidar_pts_lst = None
        if self.use_Radar:
            # radar_pts_lst = self.form_pts_lst(radar) # convert tensor into list of tensor
            # radar_pts_lst = self.form_pts_lst_no_unpad(radar) # convert tensor into list of tensor
            if mode == 'train':
                radar_pts_lst = radar # radar already is a list
            elif mode == 'test':
                radar_pts_lst = radar # changed the multi_apply so no need to squeeze the list
                # radar_pts_lst = radar[0] # TODO: check this

        
        # Extract the lidar and radar features
        if self.use_LiDAR or self.use_Radar:
            # One stage fusion
            if not self.two_stage_fusion:
                # Middle fusion after middle encoder
                # voxels_feats = self.extract_lidar_radar_feat_middle_fusion_middle_encoder(lidar_pts_lst, radar_pts_lst)

                # Middle fusion after Backbone
                voxels_feats = self.extract_lidar_radar_feat_middle_fusion(lidar_pts_lst, radar_pts_lst)

                # Middle fusion after FPN
                # voxels_feats = self.extract_lidar_radar_feat_middle_fusion_fpn(lidar_pts_lst, radar_pts_lst)
            # Two stage fusion
            else:
                # Fisrt fusion: Concatenation after middle encoder
                # Second fusion: Gated fusion after second backbone
                raise NotImplementedError
        else:
            voxels_feats = None

        # Extract the camera features
        if self.use_Cam:
            img_feats = self.extract_img_feat(img, img_metas)
        else:
            img_feats = None
        
        # Here we set radar_feats to be None
        # Actually the real radar feats is fused in voxels_feats
        return (img_feats, voxels_feats, None)

    def extract_pts_feat(self, pts, img_feats, img_metas, fused_points=None, mode='train'):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if mode == 'test':
            # print(len(fused_points))
            # print(type(fused_points[0]))
            # print(len(fused_points[0]))
            fused_points = fused_points[0]
        voxels, num_points, coors = self.voxelize(fused_points)

        
        voxel_features = self.voxel_fusion_layer(voxels, num_points, coors)

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas, fused_points=None, mode='train'):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas, fused_points=fused_points, mode=mode)
        return (img_feats, pts_feats)
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      fused_points=None,
                      radar=None,):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        # img_feats, pts_feats = self.extract_feat(
        #     points, img=img, img_metas=img_metas, fused_points=fused_points)
        assert self.middle_fusion == True
        img_feats, pts_feats, _ = self.extract_feat_v3(points, img=img, radar=radar, img_metas=img_metas, fused_points=fused_points)
        
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        return losses

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def simple_test(self, points, img_metas, img=None, rescale=False, fused_points=None, radar=None,):
        """Test function without augmentaiton."""
        # img_feats, pts_feats = self.extract_feat(
        #     points, img=img, img_metas=img_metas, fused_points=fused_points, mode='test')
        img_feats, pts_feats, _ = self.extract_feat_v3(
                points, img=img, radar=radar, img_metas=img_metas, fused_points=fused_points, mode='test')

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton.

        The function implementation process is as follows:

            - step 1: map features back for double-flip augmentation.
            - step 2: merge all features and generate boxes.
            - step 3: map boxes back for scale augmentation.
            - step 4: merge results.

        Args:
            feats (list[torch.Tensor]): Feature of point cloud.
            img_metas (list[dict]): Meta information of samples.
            rescale (bool, optional): Whether to rescale bboxes.
                Default: False.

        Returns:
            dict: Returned bboxes consists of the following keys:

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): Predicted bboxes.
                - scores_3d (torch.Tensor): Scores of predicted boxes.
                - labels_3d (torch.Tensor): Labels of predicted boxes.
        """
        # only support aug_test for one sample
        outs_list = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            # merge augmented outputs before decoding bboxes
            for task_id, out in enumerate(outs):
                for key in out[0].keys():
                    if img_meta[0]['pcd_horizontal_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[2])
                        if key == 'reg':
                            outs[task_id][0][key][:, 1, ...] = 1 - outs[
                                task_id][0][key][:, 1, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                    if img_meta[0]['pcd_vertical_flip']:
                        outs[task_id][0][key] = torch.flip(
                            outs[task_id][0][key], dims=[3])
                        if key == 'reg':
                            outs[task_id][0][key][:, 0, ...] = 1 - outs[
                                task_id][0][key][:, 0, ...]
                        elif key == 'rot':
                            outs[task_id][0][
                                key][:, 1,
                                     ...] = -outs[task_id][0][key][:, 1, ...]
                        elif key == 'vel':
                            outs[task_id][0][
                                key][:, 0,
                                     ...] = -outs[task_id][0][key][:, 0, ...]

            outs_list.append(outs)

        preds_dicts = dict()
        scale_img_metas = []

        # concat outputs sharing the same pcd_scale_factor
        for i, (img_meta, outs) in enumerate(zip(img_metas, outs_list)):
            pcd_scale_factor = img_meta[0]['pcd_scale_factor']
            if pcd_scale_factor not in preds_dicts.keys():
                preds_dicts[pcd_scale_factor] = outs
                scale_img_metas.append(img_meta)
            else:
                for task_id, out in enumerate(outs):
                    for key in out[0].keys():
                        preds_dicts[pcd_scale_factor][task_id][0][key] += out[
                            0][key]

        aug_bboxes = []

        for pcd_scale_factor, preds_dict in preds_dicts.items():
            for task_id, pred_dict in enumerate(preds_dict):
                # merge outputs with different flips before decoding bboxes
                for key in pred_dict[0].keys():
                    preds_dict[task_id][0][key] /= len(outs_list) / len(
                        preds_dicts.keys())
            bbox_list = self.pts_bbox_head.get_bboxes(
                preds_dict, img_metas[0], rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        if len(preds_dicts.keys()) > 1:
            # merge outputs with different scales after decoding bboxes
            merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, scale_img_metas,
                                                self.pts_bbox_head.test_cfg)
            return merged_bboxes
        else:
            for key in bbox_list[0].keys():
                bbox_list[0][key] = bbox_list[0][key].to('cpu')
            return bbox_list[0]

    def aug_test(self, points, img_metas, imgs=None, rescale=False, fused_points=None, radar=None):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs, fused_points=fused_points, radar=radar)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None, fused_points=None, radar=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats, radar_feats = multi_apply(self.extract_feat_v3, points, imgs,
                                           img_metas, fused_points,radar, mode='test')
        return img_feats, pts_feats