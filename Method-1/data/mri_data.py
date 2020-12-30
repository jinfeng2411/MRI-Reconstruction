"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
from torch.utils.data import Dataset
import numpy as np

class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """
    def __init__(self, root, transform):
        self.transform = transform  #从数据集中读取出 K 空间的数据后，所做的变换
        self.examples = []
        files = list(pathlib.Path(root).iterdir())  #获取 root 文件夹下的所有文件
        for fname in sorted(files):
            kspace = np.load(fname)
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(num_slices)]  #self.examples中存储的内容为：文件路径 + 该文件中每一个 slice

    def __len__(self):  # 所有文件的所有slice之和
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        data = np.load(fname)
        kspace = data[slice]
        return self.transform(kspace, fname.name, slice)
