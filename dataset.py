import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class Data(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        # root path should be:/mnt/nas/herokuma/cycle_gan_dataset/monet2photo/
        self.files_X = sorted(glob.glob(os.path.join(root, '%s/X' % mode) + '/*.*'))
        self.files_Y = sorted(glob.glob(os.path.join(root, '%s/Y' % mode) + '/*.*'))

    def __getitem__(self, index):
        item_X = self.transform(Image.open(self.files_X[index % len(self.files_X)]))

        if self.unaligned:
            item_Y = self.transform(Image.open(self.files_Y[random.randint(0, len(self.files_Y) - 1)]))
        else:
            item_Y = self.transform(Image.open(self.files_Y[index % len(self.files_Y)]))

        return {'X': item_X, 'Y': item_Y}

    def __len__(self):
        return max(len(self.files_X), len(self.files_Y))