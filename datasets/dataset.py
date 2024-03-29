from __future__ import print_function

import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from PIL import Image
# from .tsv_io import TSVFile
import numpy as np
import base64
import io
import bisect


dataset_map = {8: 5, 9: 1, 10: 1, 11: 1} # datasets-order clipart coco deeplesion dota kitchen lisa  watercolor widerface kitti voc07 voc12 comic
# dataset_map = {8: 3, 9: 6, 10: 6, 11: 1} # datasets-order clipart coco  dota kitchen lisa  watercolor widerface kitti comic voc07 voc12  deeplesion
# dataset_map = {4: 3} # datasets-order kitti clipart dota voc07 voc12 
# dataset_map = {2:1} # datasets-order  voc07 voc12
# dataset_map = {1:0, 2:1, 3: 0, 4: 0} # datasets-order  kitti clipart dota voc07 voc12

# dataset_map = {4:0} # datasets-order kitti watercolor clipart comic lisa
# dataset_map = {4:0} # datasets-order  watercolor kitti clipart  lisa comic
class SourceConcatDataset(ConcatDataset):
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        out = self.datasets[dataset_idx][sample_idx]
        if dataset_idx in dataset_map:
            dataset_idx = dataset_map[dataset_idx] # overlap voc, kitti to coco expert
        out[1].update({'dataset_idx': torch.tensor(dataset_idx)})
        return out #self.datasets[dataset_idx][sample_idx]



