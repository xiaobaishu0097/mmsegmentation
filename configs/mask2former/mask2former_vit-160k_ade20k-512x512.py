_base_ = ['./mask2former_r50_8xb2-160k_ade20k-512x512.py']
# pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_base_p16_384_20220308-96dfe169.pth'  # noqa
depths = [2, 2, 6, 2]
backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformer',
        img_size=(512, 512),
        patch_size=16,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        out_indices=(2, 5, 8, 11),
        num_heads=12,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        final_norm=True,
        norm_cfg=backbone_norm_cfg,
        with_cls_token=True,
        interpolate_mode='bicubic',
    ),
    # backbone=dict(
    #     _delete_=True,
    #     type='SwinTransformer',
    #     embed_dims=96,
    #     depths=depths,
    #     num_heads=[3, 6, 12, 24],
    #     window_size=7,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     qk_scale=None,
    #     drop_rate=0.,
    #     attn_drop_rate=0.,
    #     drop_path_rate=0.3,
    #     patch_norm=True,
    #     out_indices=(0, 1, 2, 3),
    #     with_cp=False,
    #     frozen_stages=-1,
    #     init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(in_channels=[96, 192, 384, 768]))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))
