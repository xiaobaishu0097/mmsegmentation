_base_ = ["./mask2former_r50_8xb2-160k_vegann-512x512-tvt-split1-3rd.py"]

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet101"),
    )
)

train_dataloader = dict(
    batch_size=10,
    num_workers=8,
)
