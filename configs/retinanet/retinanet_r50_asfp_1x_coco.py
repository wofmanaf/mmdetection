_base_ = './retinanet_r50_fpn_1x_coco.py'
# model settings
model = dict(
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_input',
            num_outs=5),
        dict(
            type='ASFP',
            in_channels=256,
            num_levels=5,
            refine_level=1,
            refine_type='non_local')
    ])
