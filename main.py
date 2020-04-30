import argparse
import os

import imageio
import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image
import time

from Model import SingleGenerator, SingleDiscriminator, MultipleGenerator, MultipleDiscriminator, AngRes
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

from test_multiple import test_multiple, test_angres
from test_single import test_single
from train_multiple import train_multiple, train_angres, bilinear_upsampling
from train_single import train_single
from utils import get_matlab_lf

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str,
                    default='data', help='path to dataset')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int,
                    default=4, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32,
                    help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2,
                    help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--generatorLR', type=float,
                    default=0.0001, help='learning rate for generator')
parser.add_argument('--discriminatorLR', type=float,
                    default=0.0001, help='learning rate for discriminator')
parser.add_argument('--angResLR', type=float,
                    default=0.0001, help='learning rate for angular resolution')
parser.add_argument('--cuda', action='store_true',
                    default='true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str,
                    default='checkpoints/generator_final.pth', help="path to generator weights ")
parser.add_argument('--angresWeights', type=str, default='checkpoints/AngRes_final.pth',
                    help="path to Angular resolution model weights ")
parser.add_argument('--discriminatorWeights', type=str, default='checkpoints/discriminator_final.pth',
                    help="path to discriminator weights (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints',
                    help='folder to output model checkpoints')
parser.add_argument('--continue_from', type=int, default=0,
                    help='epoch to continue training from')
parser.add_argument('--progress_images', type=bool, default=False,
                    help='saves pngs showing the current training process')
# parser.add_argument('--nima_model_path', type=str, default="checkpoints/NIMA.pth",
#                     help="path to NIMA pretrained model")

opt = parser.parse_args()
# print(opt)

writer = SummaryWriter()

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


CREATE_FOLDERS = ['output/train/bilinear_real', 'output/train/bilinear_fake', 'output/test/reals',
                  'output/test/high_res_fake', 'output/test/high_res_real', 'output/test/low_res',
                  'output/test/ang_res_fake', 'output/train/high_res_fake',
                  'output/train/high_res_real', 'output/train/low_res', 'output/train/ang_res_fake',
                  'output/train/reals', 'output/test/ang_res_fake_center',
                  'output/test/ang_res_fake_horizontaltop', 'output/test/ang_res_fake_horizontalbottom',
                  'output/test/ang_res_fake_verticalleft', 'output/test/ang_res_fake_verticalright',
                  'output/test/reals_center', 'output/test/corner1', 'output/test/corner2',
                  'output/test/corner3', 'output/test/corner4', 'output/train/ang_res_fake_center',
                  'output/train/ang_res_fake_horizontaltop', 'output/train/ang_res_fake_horizontalbottom',
                  'output/train/ang_res_fake_verticalleft', 'output/train/ang_res_fake_verticalright',
                  'output/train/reals_center', 'output/train/reals_horizontaltop',
                  'output/train/reals_horizontalbottom', 'output/train/reals_verticalright',
                  'output/train/reals_verticalleft']

try:
    for x in CREATE_FOLDERS:
        create_directory(x)
except OSError:
    print("error while creating directory")


try:
    os.makedirs(opt.out)
except OSError:
    pass

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


transform = transforms.Compose([  # transforms.CenterCrop((opt.imageSize*opt.upSampling,opt.imageSize*opt.upSampling)),
    transforms.Resize((opt.imageSize*opt.upSampling,
                       opt.imageSize*opt.upSampling)),
    transforms.ToTensor()])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
                            ])


dataset = {x: datasets.ImageFolder(os.path.join(
    opt.dataroot, x), transform=transform) for x in ['train', 'test']}

dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=opt.batchSize, drop_last=True,
                                             shuffle=False, num_workers=int(opt.workers)) for x in ['train', 'test']}

# print(len(dataset))
# generator = SingleGenerator(5, opt.upSampling)
# print(generator)
# print(len(dataset))
# discriminator = SingleDiscriminator()
# print(discriminator)

# train_single(generator, discriminator, opt, dataloader, writer, scale)

# test_single(generator, discriminator, opt, dataloader,  scale)

# generator = MultipleGenerator(16, opt.upSampling)
# print(generator)
# discriminator = MultipleDiscriminator()
# print(discriminator)
# bilinear_upsampling(opt, dataloader, scale)
# train_multiple(generator, discriminator, opt, dataloader, writer, scale)
# #
# test_multiple(generator, discriminator, opt, dataloader, scale)

ang_model = AngRes()
# ang_model.load_state_dict(torch.load(opt.angresWeights))
# print(ang_model)
#
# lfimages = get_matlab_lf()
# train_angres(ang_model, lfimages, opt, writer)
#
lfimages = get_matlab_lf(phase="test")
test_angres(ang_model, lfimages, opt)
# images = []
# import re
# numbers = re.compile(r'(\d+)')
# def numericalSort(value):
#     parts = numbers.split(value)
#     parts[1::2] = map(int, parts[1::2])
#     return parts
#
# for filename in sorted(os.listdir('output/test/high_res_fake'),key=numericalSort):
#     images.append(imageio.imread('output/test/high_res_fake/'+filename))
#
# imageio.mimsave('output/low_res.gif', images)
