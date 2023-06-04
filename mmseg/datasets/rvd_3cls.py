# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class RVD_3cls(BaseSegDataset):
    METAINFO = dict(
        classes=('class0', 'class1', 'class2'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_3cls.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
