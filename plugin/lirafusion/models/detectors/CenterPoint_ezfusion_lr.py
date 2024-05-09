import time
import torch
import numpy as np
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

from mmdet3d.models.builder import build_middle_encoder
from mmdet3d.models.builder import build_voxel_encoder

from mmcv.ops import Voxelization
from mmdet.core import multi_apply
from mmcv.runner import force_fp32
from torch.nn import functional as F





@DETECTORS.register_module()
class CenterPoint_ezfusion_lr(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                pts_voxel_layer=None,
                 pts_voxel_encoder=None,
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
                 pts_voxel_layer_radar=None,
                 pts_voxel_encoder_radar=None,
                 pts_middle_encoder_radar=None,):
        super(CenterPoint_ezfusion_lr,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)
        if pts_voxel_layer_radar:
            # Radar voxel layer
            self.pts_voxel_layer_radar = Voxelization(**pts_voxel_layer_radar)
        if pts_voxel_encoder_radar:
            # Radar encoder
            self.pts_voxel_encoder_radar = build_voxel_encoder(pts_voxel_encoder_radar)
        if pts_middle_encoder_radar:
            # Radar middle encoder
            self.pts_middle_encoder_radar = build_middle_encoder(pts_middle_encoder_radar)

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

    def extract_pts_feat(self, pts, img_feats, img_metas, radar=None, mode='train'):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None
        if mode == 'test':
            # print(len(fused_points))
            # print(type(fused_points[0]))
            # print(len(fused_points[0]))
            radar = radar
        voxels_radar, num_points_radar, coors_radar = self.voxelize_radar(radar)
        voxels_feat_radar = self.pts_voxel_encoder_radar(voxels_radar, num_points_radar, coors_radar)
        batch_size_radar = coors_radar[-1, 0] + 1
        middle_feat_radar = self.pts_middle_encoder_radar(voxels_feat_radar, coors_radar, batch_size_radar.item())
        fused_pts_lst = []
        for bs in range(batch_size_radar.item()):
            # Data Processing
            lidar_feat = pts[bs] # (N1, 5)
            radar_bev_feat = middle_feat_radar[bs, :, :, :] # (32, 128, 128)

            # Projection
            lidar_raw_coors = lidar_feat[:, :2] # (N1, 2)
            lidar_proj_coors = (lidar_raw_coors / 0.6).int() + 1 # fix this on centerpoint - voxel size is 0.6 on centerpoint
            lidar_proj_coors += int(radar_bev_feat.shape[1] / 2) - 1 # (N1, 2)

            # Concatenation
            radar_feat = radar_bev_feat[:, lidar_proj_coors[:, 0].long(), lidar_proj_coors[:, 1].long()].T # (N1, 32)
            fused_pts = torch.cat((lidar_feat, radar_feat), dim=1) # (N1, 5+32) = (N1, 37)
            fused_pts_lst.append(fused_pts)

         # Voxelize
        voxels_fused, num_points_fused, coors_fused = self.voxelize(fused_pts_lst)
        # Fused encoder
        voxels_feat_fused = self.pts_voxel_encoder(voxels_fused, num_points_fused, coors_fused)
        # Middle encoder
        # middle_feat_lidar: (bs, C*D, H, W)
        batch_size_fused = coors_fused[-1, 0] + 1
        # (B, 256, 128, 128)
        middle_feat_fused = self.pts_middle_encoder(voxels_feat_fused, coors_fused, batch_size_fused.item())
        # Go through backbone and fpn
        x = self.pts_backbone(middle_feat_fused) # list
        if self.with_pts_neck:
            x = self.pts_neck(x)

        return x

    def extract_feat(self, points, img, img_metas, radar=None, mode='train'):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas, radar=radar, mode=mode)
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
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, radar=radar)
        
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

    def simple_test(self, points, img_metas, img=None, rescale=False, radar=None):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, radar=radar, mode='test')

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

    def aug_test(self, points, img_metas, imgs=None, rescale=False, radar=None):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs, radar=radar)
        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None, radar=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs,
                                           img_metas, radar, mode='test')
        return img_feats, pts_feats