_base_ = [
    "../_base_/models/deeplabv3_r50-d8.py",
    "../_base_/datasets/vegann.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_80k.py",
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150),
)

train_dataloader = dict(
    batch_size=6,
    dataset=dict(
        img_suffix="-TVT-split1.png",
        seg_map_suffix="-TVT-split1.png",
    ),
)
val_dataloader = dict(
    dataset=dict(
        img_suffix="-TVT-split1.png",
        seg_map_suffix="-TVT-split1.png",
    )
)
test_dataloader = dict(
    dataset=dict(
        img_suffix="-TVT-split1.png",
        seg_map_suffix="-TVT-split1.png",
    )
)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=30000, val_interval=5000)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook", by_epoch=False, interval=5000, save_best="mIoU"
    ),
)
