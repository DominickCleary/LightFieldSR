import argparse

import scipy.io as sio
import os

import torch
from tensorboardX import SummaryWriter
from torchvision import datasets
from torchvision.transforms import transforms

from Model import AngRes
from test_multiple import test_angres
from train_multiple import train_angres
from utils import get_matlab_lf

writer = SummaryWriter()
try:
    os.makedirs('output/test/ang_res_fake')
    os.makedirs('output/train/ang_res_fake')
except OSError:
    print("error while creating directory")
    pass

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', type=str,
                    default='data', help='path to dataset')
parser.add_argument('--imageSize', type=int, default=64,
                    help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2,
                    help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--angResLR', type=float, default=0.0001,
                    help='learning rate for generator')
parser.add_argument('--cuda', action='store_true',
                    default='true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--angresWeights', type=str, default='checkpoints/AngRes_final.pth',
                    help="path to Angular resolution model weights ")
parser.add_argument('--out', type=str, default='checkpoints',
                    help='folder to output model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs('output/test/complete')
    os.makedirs('output/train/reals_verticalright')
    os.makedirs('output/test/reals_verticalright')

    os.makedirs('output/test/ang_res_fake_horizontaltop')
    os.makedirs('output/test/ang_res_fake_horizontalbottom')
    os.makedirs('output/test/ang_res_fake_verticalleft')
    os.makedirs('output/test/ang_res_fake_verticalright')

    os.makedirs('output/train/ang_res_fake_horizontaltop')
    os.makedirs('output/train/ang_res_fake_horizontalbottom')
    os.makedirs('output/train/ang_res_fake_verticalleft')
    os.makedirs('output/train/ang_res_fake_verticalright')

    os.makedirs('output/train/reals_horizontaltop')
    os.makedirs('output/train/reals_verticalleft')
    os.makedirs('output/train/reals_horizontalbottom')

    os.makedirs('output/test/reals_horizontaltop')
    os.makedirs('output/test/reals_verticalleft')
    os.makedirs('output/test/reals_horizontalbottom')

    os.makedirs('output/train/reals_center')
    os.makedirs('output/test/ang_res_fake_center')
    os.makedirs('output/test/reals_center')
    os.makedirs('output/train/ang_res_fake_center')
    os.makedirs('output/test/matlab')
except OSError:
    pass

ang_model = AngRes()
print(ang_model)

# Load training data
print('Load training data \n')

lfimages = get_matlab_lf('train')
# #
train_angres(ang_model, lfimages, opt, writer)

# lfimages = get_matlab_lf('test')

# test_angres(ang_model, lfimages, opt)
