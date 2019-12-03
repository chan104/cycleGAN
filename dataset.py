from torch.utils.data import Dataset
from PIL import Image
import os
import random


def make_dataset(dir):
    images = []
    for f in sorted(os.listdir(dir)):
        images.append(os.path.join(dir, f))
    return images


class Unaligned_dataset(Dataset):
    def __init__(self, rootA, rootB, transformA=None, transformB=None,
                 randomB=True):
        self.data_pathA = rootA
        self.data_pathB = rootB
        self.transformA = transformA
        self.transformB = transformB
        self.imagesA = make_dataset(self.data_pathA)
        self.imagesB = make_dataset(self.data_pathB)

        self.randomB = randomB

    def __getitem__(self, index):
        indexA = index % len(self.imagesA)
        if self.randomB:
            indexB = random.randint(0, len(self.imagesB) - 1)
        else:
            indexB = index % len(self.imagesB)
        imgA = Image.open(self.imagesA[indexA]).convert('RGB')
        imgB = Image.open(self.imagesB[indexB]).convert('RGB')
        if self.transformA is not None:
            imgA = self.transformA(imgA)
        if self.transformB is not None:
            imgB = self.transformB(imgB)
        return {'A': imgA, 'B': imgB}

    def __len__(self):
        return max(len(self.imagesA), len(self.imagesB))


class ImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.data_path = root
        self.transform = transform
        self.images = make_dataset(self.data_path)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imagesA)
