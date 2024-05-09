
####training settings
with2d = False
max_id = 10
reid_type = 'radar_fusing'
balance_sam = True
simfe = False
max_epochs = 20
load_from = None
lr=1e-04
samples_per_gpu=4
input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
workflow = [('train', 1),]


####Lidar
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
voxel_size = [0.075, 0.075, 0.2]
max_voxels=(90000, 120000)
sparse_shape=[41, 1440, 1440]
out_size_factor=8
grid_size = [1440, 1440, 40]
eval_depth_range = None #[30,40]
eval_ve_range = None #[2, float('inf')]  #[0, 3] [3,, 10] [10, float('inf')] 

norm_cfg = dict(type='BN', requires_grad=True)

####class setting
class_names = [
    'car', 'truck',  'bus', 'trailer', 
    'motorcycle', 'bicycle', 'pedestrian', ]
task_head = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=1, class_names=['truck']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=1, class_names=['pedestrian', ])]
label_head = [1, 1, 2, 2, 1]



# If point cloud range is changed, the models should also change their point
# cloud range accordingly
voxel_size = [0.075, 0.075, 0.2]
point_cloud_range = [-54, -54, -5.0, 54, 54, 3.0]
# For nuScenes we usually do 10-class detection
# class_names = [
#     'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
#     'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
# ]
# class_names = [
#     'car', 'truck',  'bus', 'trailer', 
#     'motorcycle', 'bicycle', 'pedestrian', ] # change to 7-class

model = dict(
    type='CenterPoint',
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=[0.075, 0.075, 0.2],
        max_voxels=(90000, 120000),
        point_cloud_range=[-54, -54, -5.0, 54, 54, 3.0]),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=512,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=1, class_names=['truck']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=1, class_names=['pedestrian'])
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            code_size=9,
            pc_range=[-54, -54]),
        separate_head=dict(
            type='DCNSeparateHead',
            init_bias=-2.19,
            final_kernel=3,
            norm_cfg=dict(type='BN', requires_grad=True),
            dcn_config=dict(
                type='DCN',
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
                groups=4)),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            grid_size=[1440, 1440, 40],
            voxel_size=[0.075, 0.075, 0.2],
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=[-54, -54, -5.0, 54, 54, 3.0])),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=[0.075, 0.075],
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2,
            pc_range=[-54, -54])))

# dataset_type = 'NuScenesDataset'
dataset_type = 'NuScenesDatasetRadar' # test if this data loader has issue
data_root = '/scratch/kskin_root/kskin1/jingyuso/nuScenes/'
# data_root = '/mnt/ws-frb/users/jingyuso/dataset/nuScenes_mini/'

file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        bus=4,
        trailer=6,
        motorcycle=6,
        bicycle=6,
        pedestrian=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=True,
        pcd_horizontal_flip=True,
        pcd_vertical_flip=True,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', sync_2d=False),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

# add what we got from ezfusion
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args,
        pad_empty_sweeps=True,
        remove_close=True,
        lid_range=point_cloud_range,),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        pcd_horizontal_flip=True,
        pcd_vertical_flip=True,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D', sync_2d=False),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
evaluation = dict(
    interval=20,
    pipeline=eval_pipeline)


data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=samples_per_gpu,
    train=dict(
        type='CBGSDataset',
        data_root=data_root,
        ann_file=data_root + 'radar_nuscenes_5sweeps_infos_train_radar_coor_1.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=file_client_args),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox=True,
                with_label=True,
                with_attr_label=True,
                with_bbox_depth=True),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=point_cloud_range),
            dict(
                type='ObjectNameFilter',
                classes=class_names),
            dict(type='PointShuffle'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names),
            dict(
                type='Collect3D',
                keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ],
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        dataset=dict(
            type='NuScenesDataset',
            data_root=data_root,
            ann_file=data_root + 'radar_nuscenes_5sweeps_infos_train_radar_coor_1.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR'),
        box_type_3d='LiDAR'),
    val=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'radar_nuscenes_5sweeps_infos_val_radar_coor_1.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=data_root + 'radar_nuscenes_5sweeps_infos_val_radar_coor_1.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))

optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, lr),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
log_level = 'INFO'
resume_from = None
dist_params = dict(backend='nccl')

find_unused_parameters = True
