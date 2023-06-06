_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/drive_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2))

train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=2000,
        save_best='mIoU'),)
