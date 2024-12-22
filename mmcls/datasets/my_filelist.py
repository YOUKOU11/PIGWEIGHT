from typing import Optional, Sequence, Union
import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PigNet(BaseDataset):
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')
    # CLASSES = [
    #     'back_and_forth',
    #     'fast',
    #     'human_or_still-life',
    #     'jump',
    #     'lame',
    #     'normal',
    #     'slow',
    #     'uncooperative']
    CLASSES = [
        'fast',
        'normal',
        'slow',
        'uncooperative']
    # CLASSES = [
    #     'back_and_forth',
    #     'human_or_still-life',
    #     'lame',
    #     'normal'
    # ]

    def load_annotations(self):
        assert isinstance(self.ann_file, str)
        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(',') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix, 'img_info': {'filename': filename},
                        'gt_label': np.array(gt_label, dtype=np.int64)}
                data_infos.append(info)
            return data_infos
