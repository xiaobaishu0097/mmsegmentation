# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class RVD_9cls(BaseSegDataset):
    METAINFO = dict(
        classes=('class0', 'class1', 'class2', 'class3', 'class4', 'class5',
                 'class6', 'class7', 'class8'),
        palette=[
            [120, 120, 120],
            [180, 120, 120],
            [6, 230, 230],
            [80, 50, 50],
            [4, 200, 3],
            [120, 120, 80],
            [140, 140, 140],
            [204, 5, 255],
            [230, 230, 230],
        ])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_9cls.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
