_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/rvd_3cls_256x256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=3,),
    auxiliary_head=dict(
        num_classes=3,),
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))
