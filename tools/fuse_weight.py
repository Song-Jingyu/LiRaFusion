import torch

# camera_ckpt_path = '/mnt/workspace/users/pinghual/cam_only/epoch_24.pth'
camera_radar_ckpt_path = '/mnt/workspace/users/pinghual/cam_only/epoch_24.pth'
lidar_checkpoint_path = '/mnt/workspace/users/jingyuso/fusion_radar/purrgil/KP-FUTR3D/work_dirs/lidar_radar_voxel_fusion_aug/epoch_42.pth'
# lidar_cam_ckpt_path = '/mnt/workspace/users/jingyuso/KP-FUTR3D/work_dirs/res101_01voxel_step_3e/epoch_3.pth'
saved_path = './pretrained/lidar_radar_cam_new.pth'

img_ckpt = torch.load(camera_radar_ckpt_path)
state_dict1 = img_ckpt['state_dict']

pts_ckpt = torch.load(lidar_checkpoint_path)
state_dict2 = pts_ckpt['state_dict']
# pts_head in camera checkpoint will be overwrite by lidar checkpoint
state_dict1.update(state_dict2)

merged_state_dict = state_dict1

save_checkpoint = {'state_dict':merged_state_dict }

torch.save(save_checkpoint, saved_path)