#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from functools import partial
from tqdm.contrib.concurrent import process_map

from ..segment_base import SegmentBase
from ..utils.segmentation_utils import label_graph


# Segmentation data module specific to the TrackML pipeline
class TrackMLSegment(SegmentBase):
    def __init__(self, hparams):
        super().__init__(hparams)


    def prepare_data(self):

        all_files = [
            os.path.join(self.hparams["input_dir"], file)
            for file in os.listdir(self.hparams["input_dir"])
        ][: self.n_files]
        all_files = np.array_split(all_files, self.n_tasks)[self.task]

        os.makedirs(self.output_dir, exist_ok=True)
        print("Writing outputs to " + self.output_dir)

        process_func = partial(label_graph, **self.hparams)
        process_map(process_func, all_files, max_workers=self.n_workers)
