_base_ = "./deeplabv3_r50-d8_4xb4-80k_vegann-512x512-tvt-split3-2nd.py"
model = dict(pretrained="open-mmlab://resnet101_v1c", backbone=dict(depth=101))
