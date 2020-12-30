"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
from pathlib import Path
import pathlib
import scipy.io as sio
import numpy as np
import torch

import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
from data.transforms import to_complex
from utils.utils import save_reconstructions
from collections import defaultdict

def save_zero_filled(data_dir, out_dir):
    mask = sio.loadmat("./mask/%s/%s/%s_256_256_%d.mat" % (args.data, args.mask, args.mask, args.rate))['Umask']
    for tgt_file in args.data_path.iterdir():
        reconstructions = defaultdict(list)
        print(out_dir/tgt_file.name)
        tgt_kspace = np.load(tgt_file)  # K-space, (num_slice,256,256,2)

        under_img = np.zeros(tgt_kspace.shape[:3])
        #print(tgt_img.shape)  # (num_slices,256,256)
        for i in range(len(tgt_kspace)):
            # K空间欠采样
            tgt_kspace_i = tgt_kspace[i, :, :, :]
            under_kspace_i = np.zeros_like(tgt_kspace_i)
            under_kspace_i[:,:,0] = np.fft.ifftshift(np.where(mask == 0, 0,np.fft.fftshift(tgt_kspace_i[:,:,0])))   #保留大部分低频成分
            under_kspace_i[:,:,1] = np.fft.ifftshift(np.where(mask == 0, 0,np.fft.fftshift(tgt_kspace_i[:,:,1])))   #保留大部分低频成分
            under_kspace_tensor_i = torch.from_numpy(under_kspace_i)

            under_img_tensor_i = torch.ifft(under_kspace_tensor_i, 2, normalized=True)
            under_img_i = under_img_tensor_i.numpy()
            under_img_i = np.abs(to_complex(under_img_i))

            under_img[i,:,:] = under_img_i

        reconstructions = {
            tgt_file.name: under_img
        }

        save_reconstructions(reconstructions, out_dir)


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-path", type=Path, required=True,
        help="Path to save the reconstructions to")
    parser.add_argument('--data', type=str, default='brain', choices=['brain', 'knee'],
                    help='which dataset')
    parser.add_argument('--rate', type=int, default=20, choices=[5, 10, 20, 25],
                    help='The undersampling rate')
    parser.add_argument('--mask', type=str, default='radial', choices=['cartesian', 'radial', 'random'],
                    help='The type of mask')
    parser.add_argument('--data-path', type=pathlib.Path, required=True,
                    help='Path to the dataset')

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    save_zero_filled(args.data_path, args.out_path)
