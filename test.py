import time
import torch
import torchvision.transforms as transforms
import numpy as np
import os.path
import utils
from cycleGAN import cycleGAN
from tqdm import tqdm

import dataset
import argparse


def dataloader(batch_size, data_root, shuffle=False, num_workers=8):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testset = dataset.ImageFolder(data_root, transform_test)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers)

    return testloader


def run_one_epoch(net, loader, training, max_length=10000):
    loss = []
    for idx, data in enumerate(loader):
        if (idx >= max_length):
            break
        with torch.set_grad_enabled(training):
            net.process_one_input(data, training)
        loss.append(net.get_loss())
    loss = np.array(loss)
    if training:
        model.schedulers_step()
    return np.mean(loss, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--save_prefix', default="project_test")
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--mode', default="A2B")

    parser.add_argument('--load_model', required=True)

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    Logger = utils.Logger(os.path.join(
        args.save_path, args.save_prefix + '.log'))

    Logger(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger("using device: ", device)

    data_root = args.dataroot
    batch_size = args.batch_size
    testloader = dataloader(batch_size, data_root)

    model = torch.load(args.load_model)
    model.mode = args.mode
    model.to(device)

    for idx, data in tqdm(enumerate(testloader)):
        if idx * batch_size > args.max_length:
            break
        with torch.set_grad_enabled(False):
            model.one_side_process(data)
        model.save_current_batch(
            args.save_path, args.mode, idx * batch_size)

    Logger("done")
    Logger(time.asctime())
