import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def savefig(img, path):
    img = (img + 1) / 2
    img = np.moveaxis(img, 0, 2)
    mpimg.imsave(path + '.png', img, format='png')


def plot(img):
    img = (img + 1) / 2
    plt.imshow(np.moveaxis(img, 0, 2))
    plt.show()


class Logger():
    def __init__(self, file_name):
        self.log_file = open(file_name, 'a')

    def __call__(self, *args):
        print(*args)
        print(*args, file=self.log_file, flush=True)


class Linear_decay(object):
    # To fix "unable to pickle lambda" problem
    def __init__(self, L=50):
        self.L_1 = L
        self.L_2 = L

    def __call__(self, epoch):
        return 1 if epoch < self.L_1 else (
            self.L_1 + self.L_2 - epoch) / self.L_2
