import logging
import pathlib
import shutil
import time
import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader
import scipy.io as sio
import argparse
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from data.mri_data import SliceData
from data.transforms import to_complex
from utils.vis_tools import Visualizer
from unet_model import UnetModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
        
        return  torch.from_numpy(under_img),torch.from_numpy(target_img)


def create_datasets(args):
    mask = sio.loadmat("./mask/%s/%s/%s_256_256_%d.mat" % (args.data, args.mask, args.mask, args.rate))['Umask']
    train_data = SliceData(
        root = args.data_path/f'Train',
        transform = DataTransform(mask),
    )
    dev_data = SliceData(
        root = args.data_path/f'Val',
        transform = DataTransform(mask),
    )
    return dev_data,train_data

def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0,len(dev_data),len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset = train_data,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = 4,
        pin_memory = True,
    )
    dev_loader = DataLoader(
        dataset = dev_data,
        batch_size = args.batch_size,
        num_workers = 4,
        pin_memory = True,
    )

    display_loader = DataLoader(
        dataset = display_data,
        batch_size = 16,
        num_workers = 4,
        pin_memory = True,
    )
    return train_loader,dev_loader,display_loader

def train_epoch(args, epoch, model, data_loader, optimizer, writer):
    model.train()
    avg_loss = 0.
    ##每一个 epoch 开始的时间 和 每一个 iter 开始的时间（初始值相同）
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        under_img_tensor, target_img_tensor = data
        under_img_tensor = under_img_tensor.unsqueeze(1).to(args.device)
        target_img_tensor = target_img_tensor.to(args.device)

        output = model(under_img_tensor).squeeze(1)
        loss = F.l1_loss(output, target_img_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer, vis):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            under_img_tensor, target_img_tensor = data
            under_img_tensor = under_img_tensor.unsqueeze(1).to(args.device)
            target_img_tensor = target_img_tensor.to(args.device)

            output = model(under_img_tensor).squeeze(1)
            loss = F.mse_loss(output, target_img_tensor)
            losses.append(loss.item())
        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
        if(vis!=None):
            vis.plot("Val Loss",np.mean(losses))

    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer, vis):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            under_img_tensor, target_img_tensor = data
            under_img_tensor = under_img_tensor.unsqueeze(1).to(args.device)
            target_img_tensor = target_img_tensor.to(args.device)
            output = model(under_img_tensor)

            if(vis!=None):
                for i in range(len(output)):
                    vis.img("Undersampled image - %d"%(i), under_img_tensor.squeeze(1)[i])
                    vis.img("Full image - %d"%(i), target_img_tensor[i])
                    vis.img("Recons image - %d"%(i), output[i])
            save_image(target_img_tensor.unsqueeze(1), 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target_img_tensor.unsqueeze(1) - output), 'Error')
            break


def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    model = UnetModel(1, 1, 64, 4, 0).to(args.device)
    return model


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model = build_model(args)
    model.load_state_dict(checkpoint['model'])
    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, model, optimizer


def build_optim(args, params):
    #optimizer = torch.optim.RMSprop(params, args.lr)
    optimizer = torch.optim.Adam(params, args.lr)
    return optimizer


def main(args, vis):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    if args.resume:
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model = build_model(args)
        optimizer = build_optim(args, model.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)

    train_loader, dev_loader, display_loader = create_data_loaders(args)

    for epoch in range(start_epoch, args.epochs):
        #scheduler.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, writer)

        dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer, vis)
        visualize(args, epoch, model, display_loader, writer, vis)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
        #scheduler.step(epoch)
    writer.close()


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', default=12, type=int,
                        help='Mini batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='The learning rate')
    parser.add_argument('--rate', type=int, default=20, choices=[5, 10, 20, 25],
                        help='The undersampling rate')
    parser.add_argument('--mask', type=str, default='radial', choices=['cartesian', 'radial', 'random'],
                        help='The type of mask')
    parser.add_argument('--data', type=str, default='brain', choices=['brain', 'knee'],
                        help='which dataset')
    parser.add_argument('--report-interval', type=int, default=100,
                        help='Period of loss reporting')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--data-path', type=pathlib.Path, required=True,
                        help='Path to the dataset')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Which device to train on. Set to "cuda:n" ,n represent the GPU number')
    parser.add_argument('--use-visdom',type=bool,default=False,
                        help='if True, watch loss and reconstructions on port http://localhost:8097'
                        )
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    if(args.use_visdom):
        vis = Visualizer()
    else:
        vis = None
    main(args,vis)
