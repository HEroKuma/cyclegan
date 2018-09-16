import argparse
import os
import numpy as np
import math
import sys
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision import datasets
from torch.autograd import Variable

from model import *
from dataset import *
from utils import *

import torch

os.makedirs('images', exist_ok=True)
os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="monet2photo", help='name of the dataset')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between saving model checkpoints')
parser.add_argument('--gpu_num', type=str, default='1', help='number of gpu device')
parser.add_argument('--log_file', type=str, default='log', help='log file saved path')
parser.add_argument('--tfboard', type=bool, default=True, help='use tensorboard or not')
opt = parser.parse_args()
print(opt)

os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_num
tf_iter = 0
log_size = 100

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2**4, opt.img_width // 2**4)

# Initialize generator and discriminator
G_XY = ResNet()
G_YX = ResNet()
D_X = Discriminator()
D_Y = Discriminator()

root = '/mnt/nas/herokuma/cycle_gan_dataset/'

if cuda:
    G_XY = G_XY.cuda()
    G_YX = G_YX.cuda()
    D_X = D_X.cuda()
    D_Y = D_Y.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_XY.load_state_dict(torch.load('saved_models/G_XY_%d.pth'))
    G_YX.load_state_dict(torch.load('saved_models/G_YX_%d.pth'))
    D_X.load_state_dict(torch.load('saved_models/D_X_%d.pth'))
    D_Y.load_state_dict(torch.load('saved_models/D_Y_%d.pth'))
else:
    # Initialize weights
    G_XY.apply(weights_init_normal)
    G_YX.apply(weights_init_normal)
    D_X.apply(weights_init_normal)
    D_Y.apply(weights_init_normal)

# Loss weights
lambda_cyc = 10
lambda_id = 0.5 * lambda_cyc

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_XY.parameters(), G_YX.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_X = torch.optim.Adam(D_X.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_Y = torch.optim.Adam(D_Y.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_X = torch.optim.lr_scheduler.LambdaLR(optimizer_D_X, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_Y = torch.optim.lr_scheduler.LambdaLR(optimizer_D_Y, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_X_buffer = ReplayBuffer()
fake_Y_buffer = ReplayBuffer()

# Image transformations
transforms_ = [ transforms.Resize(int(opt.img_height*1.12), Image.BICUBIC),
                transforms.RandomCrop((opt.img_height, opt.img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

# Training data loader
dataloader = DataLoader(Data(os.path.join(root, opt.dataset_name), transforms_=transforms_, unaligned=True),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
# Test data loader
val_dataloader = DataLoader(Data(os.path.join(root, opt.dataset_name), transforms_=transforms_, unaligned=True, mode='test'),
                        batch_size=5, shuffle=True, num_workers=1)
# TFboard data loader
test_loader = DataLoader(Data(os.path.join(root, opt.dataset_name), transforms_=transforms_, unaligned=True, mode='test'),
                        batch_size=1, shuffle=True, num_workers=1)


# def loss_tfboard(data_loader, writer: SummaryWriter = None, epoch=None):
#     X = next(iter(data_loader))['X']
#     Y = next(iter(data_loader))['Y']
#
#     if epoch == 0:


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    real_X = Variable(imgs['X'].type(Tensor))
    fake_Y = G_XY(real_X)
    real_Y = Variable(imgs['Y'].type(Tensor))
    fake_X = G_YX(real_Y)
    img_sample = torch.cat((real_X.data, fake_Y.data,
                            real_Y.data, fake_X.data), 0)
    save_image(img_sample, '/mnt/nas/herokuma/cycle_gan_dataset/gen_img/%s.png' % batches_done, nrow=5, normalize=True)

# ----------
#  Training
# ----------

start_time = time.time()
if opt.tfboard:
    writer_G = SummaryWriter('log/G')
    writer_D_X = SummaryWriter('log/D/X')
    writer_D_Y = SummaryWriter('log/D/Y')
else:
    writer_G = None
    writer_D_X = None
    writer_D_Y = None


for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_X = Variable(batch['X'].type(Tensor))
        real_Y = Variable(batch['Y'].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_X.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_X.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_X = criterion_identity(G_YX(real_X), real_X)
        loss_id_Y = criterion_identity(G_XY(real_Y), real_Y)

        loss_identity = (loss_id_X + loss_id_Y) / 2

        # GAN loss
        fake_Y = G_XY(real_X)
        loss_GAN_XY = criterion_GAN(D_Y(fake_Y), valid)
        fake_X = G_YX(real_Y)
        loss_GAN_YX = criterion_GAN(D_X(fake_X), valid)

        loss_GAN = (loss_GAN_XY + loss_GAN_YX) / 2

        # Cycle loss
        recov_X = G_YX(fake_Y)
        loss_cycle_X = criterion_cycle(recov_X, real_X)
        recov_Y = G_XY(fake_X)
        loss_cycle_Y = criterion_cycle(recov_Y, real_Y)

        loss_cycle = (loss_cycle_X + loss_cycle_Y) / 2

        # Total loss
        loss_G =    loss_GAN + \
                    lambda_cyc * loss_cycle + \
                    lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()
        if i % log_size == 0:
            G_loss_info = {
                'identity_loss_X': loss_id_X,
                'identity_loss_Y': loss_id_Y,
                'identity_loss': loss_identity,
                'GAN_loss_XY': loss_GAN_XY,
                'GAN_loss_YX': loss_GAN_YX,
                'GAN_loss': loss_GAN,
                'cycle_loss_X': loss_cycle_X,
                'cycle_loss_Y': loss_cycle_Y,
                'cycle_loss': loss_cycle,
                'total_loss': loss_G,
            }
        # -----------------------
        #  Train Discriminator X
        # -----------------------

        optimizer_D_X.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_X(real_X), valid)
        # Fake loss (on batch of previously generated samples)
        fake_X_ = fake_X_buffer.push_and_pop(fake_X)
        loss_fake = criterion_GAN(D_X(fake_X_.detach()), fake)
        # Total loss
        loss_D_X = (loss_real + loss_fake) / 2

        loss_D_X.backward()
        optimizer_D_X.step()

        if i % log_size == 0:
            D_X_info = {
                'real_loss' : loss_real,
                'fake_loss' : loss_fake,
                'D_X_loss' : loss_D_X,
            }
        # -----------------------
        #  Train Discriminator Y
        # -----------------------

        optimizer_D_Y.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_Y(real_Y), valid)
        # Fake loss (on batch of previously generated samples)
        fake_Y_ = fake_Y_buffer.push_and_pop(fake_Y)
        loss_fake = criterion_GAN(D_Y(fake_Y_.detach()), fake)
        # Total loss
        loss_D_Y = (loss_real + loss_fake) / 2

        loss_D_Y.backward()
        optimizer_D_Y.step()

        loss_D = (loss_D_X + loss_D_Y) / 2

        if i % log_size == 0:
            D_Y_info = {
                'real_loss': loss_real,
                'fake_loss': loss_fake,
                'D_Y_loss': loss_D_Y,
            }
        # --------------
        #  Add loss log to TFboard
        # --------------
        if i % log_size == 0:
            if writer_G and writer_D_X and writer_D_Y:
                tf_iter
                for key, value in G_loss_info.items():
                    writer_G.add_scalar(key, value.item(), tf_iter)

                for key, value in D_X_info.items():
                    writer_D_X.add_scalar(key, value.item(), tf_iter)

                for key, value in D_Y_info.items():
                    writer_D_Y.add_scalar(key, value.item(), tf_iter)
                tf_iter += 1

        # if writer:
        #     loss_tfboard(test_loader, writer, epoch)

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time)/ (batches_done + 1))

        # Print log
        print("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_cycle.item(),
                                                        loss_identity.item(), time_left))

        # If at sample interval save image
        #if batches_done % opt.sample_interval == 0:
         #   writer.add_scalar('loss', )
        #     sample_images(batches_done)
        sample_images(epoch)


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_X.step()
    lr_scheduler_D_Y.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_XY.state_dict(), 'saved_models/G_AB_%d.pth' % epoch)
        torch.save(G_YX.state_dict(), 'saved_models/G_BA_%d.pth' % epoch)
        torch.save(D_X.state_dict(), 'saved_models/D_A_%d.pth' % epoch)
        torch.save(D_Y.state_dict(), 'saved_models/D_B_%d.pth' % epoch)
