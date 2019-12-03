import matplotlib.pyplot as plt
import numpy as np


def savefig(img, path):
    img = (img + 1) / 2
    img = np.moveaxis(img, 0, 2)
    plt.imshow(img)
    plt.savefig(path)


def plot(img):
    img = (img + 1) / 2
    plt.imshow(np.moveaxis(img, 0, 2))
    plt.show()


class Logger():
    def __init__(self, file_name):
        self.log_file = open(file_name, 'w')

    def __call__(self, *args):
        print(*args)
        print(*args, file=self.log_file, flush=True)
