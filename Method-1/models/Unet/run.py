import numpy as np
import torch
from torch.utils.data import DataLoader
import pathlib
from collections import defaultdict
import scipy.io as sio
import argparse
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from data.mri_data import SliceData
from data.transforms import to_complex
from unet_model import UnetModel
from utils.utils import save_reconstructions


#从每一张图像的原始数据（即 频域复数）中，
# 得到 target_kspace_tensor，target_img_tensor，
# under_kspace_tensor，under_img_tensor
class DataTransform:
    def __init__(self, mask):
        self.mask = torch.from_numpy(mask)

    def __call__(self, kspace, fname, slice):
        target_kspace = kspace
        under_kspace = np.zeros_like(target_kspace)
        under_kspace[:,:,0] = np.fft.ifftshift(np.where(self.mask == 0, 0,np.fft.fftshift(target_kspace[:,:,0])))   #保留大部分低频成分
        under_kspace[:,:,1] = np.fft.ifftshift(np.where(self.mask == 0, 0,np.fft.fftshift(target_kspace[:,:,1])))   #保留大部分低频成分

        target_kspace_tensor = torch.from_numpy(target_kspace).type(torch.FloatTensor)
        under_kspace_tensor = torch.from_numpy(under_kspace).type(torch.FloatTensor)
        target_img_tensor = torch.ifft(target_kspace_tensor, 2, normalized=True).type(torch.FloatTensor)
        under_img_tensor = torch.ifft(under_kspace_tensor, 2, normalized=True).type(torch.FloatTensor)

        target_img = target_img_tensor.numpy()
        under_img = under_img_tensor.numpy()

        # for
        target_img = np.abs(to_complex(target_img))
        under_img = np.abs(to_complex(under_img))

        target_img = (target_img-np.min(target_img))/(np.max(target_img)-np.min(target_img))*255 - 128
        under_img = (under_img-np.min(under_img))/(np.max(under_img)-np.min(under_img))*255 - 128

        return  torch.from_numpy(under_img),torch.from_numpy(target_img),fname,slice



def create_data_loaders(args):
    mask = sio.loadmat("./mask/%s/%s/%s_256_256_%d.mat" % (args.data, args.mask, args.mask, args.rate))['Umask']
    data = SliceData(
        root = args.data_path/f'Val',
        transform = DataTransform(mask),
    )
    data_loader = DataLoader(
        dataset=data,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    return data_loader

#导入由 train.py 训练好的模型
#load the model generate by train.py
def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = UnetModel(1, 1, 64, 4, 0).to(args.device)
    model.load_state_dict(checkpoint['model'])
    return model

#将 Val 中的数据欠采样后送入网络，重建图像以相同的文件名、numpy数组形式保存
def run_unet(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(list)
    with torch.no_grad():
        for data in data_loader:
            under_img_tensor, target_img_tensor, fnames, slices = data
            under_img_tensor = under_img_tensor.unsqueeze(1).to(args.device)

            recons = model(under_img_tensor).to('cpu').squeeze(1)
            for i in range(recons.shape[0]):
                reconstructions[fnames[i]].append((slices[i].numpy(), recons[i].numpy()))

    reconstructions = {
        fname: np.stack([pred for _, pred in sorted(slice_preds)])
        for fname, slice_preds in reconstructions.items()
    }
    return reconstructions


def main(args):
    data_loader = create_data_loaders(args)
    model = load_model(args.checkpoint)
    reconstructions = run_unet(args, model, data_loader)
    save_reconstructions(reconstructions, args.out_dir)
    print("write the reconstructions of Val to {} successfully!".format(str(args.out_dir)))


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='brain', choices=['brain', 'knee'],
                        help='which dataset')
    parser.add_argument('--rate', type=int, default=20, choices=[5, 10, 20, 25],
                        help='The undersampling rate')
    parser.add_argument('--mask', type=str, default='radial', choices=['cartesian', 'radial', 'random'],
                        help='The type of mask')
    parser.add_argument('--data-path', type=pathlib.Path, required=True,
                        help='Path to the dataset')
    parser.add_argument('--checkpoint', type=pathlib.Path, required=True,
                        help='Path to the U-Net model')
    parser.add_argument('--out-dir', type=pathlib.Path, required=True,
                        help='Path to save the reconstructions to')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='Mini-batch size')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Which device to train on. Set to "cuda:n" ,n represent the GPU number')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    main(args)
