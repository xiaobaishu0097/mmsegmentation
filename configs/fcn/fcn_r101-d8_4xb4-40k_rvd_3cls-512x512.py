_base_ = './fcn_r50-d8_4xb4-40k_rvd_3cls-512x512.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))