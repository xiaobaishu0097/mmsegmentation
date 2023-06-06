_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', '../_base_/datasets/chase_db1_256x256.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(crop_size=(256, 256), stride=(170, 170)))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=2000,
        save_best='mIoU'),)