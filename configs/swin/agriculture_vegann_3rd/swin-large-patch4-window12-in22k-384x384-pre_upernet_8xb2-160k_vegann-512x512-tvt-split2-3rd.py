_base_ = ["swin-large-patch4-window7-in22k-pre_upernet_8xb2-160k_vegann-512x512-tvt-split2-3rd.py"]
checkpoint_file = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth"  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type="Pretrained", checkpoint=checkpoint_file),
        pretrain_img_size=384,
        window_size=12,
    )
)
