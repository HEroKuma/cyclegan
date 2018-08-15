import argparse
import numpy as np
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from model import ResNet, Discriminator
from dataset import Data
from utils import ReplayBuffer, LambdaLR, weights_init_normal

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
opt = parser.parse_args()
print(opt)

# Loss
criterion_GAN = torch.nn.MSELoss() # gan loss
criterion_cycle = torch.nn.L1Loss() # shape loss for x&X y&Y
criterion_identity = torch.nn.L1Loss() # shape loss for X'&DX() Y'&DY()

cuda = True if torch.cuda.is_available() else False

# Patch size for discriminator(PatchGAN)
patch = (1, opt.img_height // 2**4, opt.img_width // 2**4)

# Init
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
    # load par
    G_XY.load_state_dict(torch.load('saved_models/G_XY_%d.pth'))
    G_YX.load_state_dict(torch.load('saved_models/G_YX_%d.pth'))
    D_X.load_state_dict(torch.load('saved_models/D_X_%d.pth'))
    D_Y.load_state_dict(torch.load('saved_models/D_Y_%d.pth'))
else:
    # or init
    G_XY.apply(weights_init_normal)
    G_YX.apply(weights_init_normal)
    D_X.apply(weights_init_normal)
    D_Y.apply(weights_init_normal)

# Loss weights
lambda_cyc = 10
lambda_id = 0.5 * lambda_cyc

# Optimizer
optimizer_G = torch.optim.Adam(itertools.chain(G_XY.parameters(), G_YX.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_X = torch.optim.Adam(D_X.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_Y = torch.optim.Adam(D_Y.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update
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


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    real_X = Variable(imgs['X'].type(Tensor))
    fake_Y = G_XY(real_X)
    real_Y = Variable(imgs['Y'].type(Tensor))
    fake_X = G_YX(real_Y)
    img_sample = torch.cat((real_X.data, fake_Y.data,
                            real_Y.data, fake_X.data), 0)
    save_image(img_sample, 'images/%s.png' % batches_done, nrow=5, normalize=True)

# train
start_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        real_X = Variable(batch['X'].type(Tensor))
        real_Y = Variable(batch['Y'].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_X.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_X.size(0), *patch))), requires_grad=False)

        # Train G
        optimizer_G.zero_grad()

        # Identity loss
        loss_id_X = criterion_identity(G_YX(real_X), real_X) # L1 norm(X, X')
        loss_id_Y = criterion_identity(G_XY(real_Y), real_Y) # L1 norm(Y, Y')

        loss_identity = (loss_id_X + loss_id_Y) / 2

        # GAN loss
        fake_Y = G_XY(real_X) # Y' = G(X)
        loss_GAN_XY = criterion_GAN(D_Y(fake_Y), valid) # identify Y'
        fake_X = G_YX(real_Y) # X' = F(Y)
        loss_GAN_YX = criterion_GAN(D_X(fake_X), valid) # identify X'

        loss_GAN = (loss_GAN_XY + loss_GAN_YX) / 2

        # Cycle loss
        recov_X = G_YX(fake_Y) # x = F(Y') = F(G(X))
        loss_cycle_X = criterion_cycle(recov_X, real_X) # x-X
        recov_Y = G_XY(fake_X) # y = G(X') = G(F(Y))
        loss_cycle_Y = criterion_cycle(recov_Y, real_Y) # y-Y

        loss_cycle = (loss_cycle_X + loss_cycle_Y) / 2

        # Total loss
        loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # ====================================================        
        
        # Train DX
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

        # ====================================================
        
        # Train DY
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

        # ====================================================

        # Log progress
        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time)/ (batches_done + 1))

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_cycle.item(),
                                                        loss_identity.item(), time_left))

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)


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
