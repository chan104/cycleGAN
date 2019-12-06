import time
import torch
import torchvision.transforms as transforms
import numpy as np
import os.path
import utils
from cycleGAN import cycleGAN

import dataset
from PIL import Image
from tqdm import tqdm
import argparse


def dataloader(batch_size, data_root, shuffle=False, num_workers=8):
    transform_train = transforms.Compose([
        transforms.Resize(286, Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = dataset.Unaligned_dataset(
        data_root + '/trainA', data_root + '/trainB',
        transform_train, transform_train)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers)

    testset = dataset.Unaligned_dataset(
        data_root + '/testA', data_root + '/testB',
        transform_test, transform_test, randomB=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers)

    return trainloader, testloader


def run_one_epoch_tqdm(net, loader, training, max_length=10000):
    loss = []
    for idx, data in tqdm(enumerate(loader)):
        if (idx >= max_length):
            break
        with torch.set_grad_enabled(training):
            net.process_one_input(data, training)
        loss.append(net.get_loss())
    loss = np.array(loss)
    if training:
        model.schedulers_step()
    return np.mean(loss, axis=0)


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
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lambda_A', type=float, default=10.0)
    parser.add_argument('--lambda_B', type=float, default=10.0)
    parser.add_argument('--lambda_idt', type=float, default=0.5)
    parser.add_argument('--save_path', default="./")
    parser.add_argument('--save_prefix', default="project")
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--start_epochs', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=10000)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--img_freq', type=int, default=5)

    parser.add_argument('--scheduler_step_size', type=int, default=10)
    parser.add_argument('--scheduler_gamma', type=float, default=0.3)

    parser.add_argument('--load_model', default=None)

    parser.add_argument('--reset_optimizer', action='store_true')

    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--g_blocks', type=int, default=9)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--use_dropout', action='store_true')
    parser.add_argument('--g_n_downsampling', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)

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
    trainloader, testloader = dataloader(batch_size, data_root)

    if args.load_model is not None:
        model = torch.load(args.load_model)
        if args.reset_optimizer:
            model.reset_optimizer(lr=args.lr, beta1=args.beta1,
                                  scheduler_gamma=args.scheduler_gamma,
                                  scheduler_step_size=args.scheduler_step_size)
    else:
        model = cycleGAN(3, 3, device, lambda_A=args.lambda_A,
                         lambda_B=args.lambda_B,
                         lambda_idt=args.lambda_idt,
                         lr=args.lr, beta1=args.beta1,
                         scheduler_gamma=args.scheduler_gamma,
                         scheduler_step_size=args.scheduler_step_size,
                         ngf=args.ngf, ndf=args.ndf, g_blocks=args.g_blocks,
                         n_layers=args.n_layers, use_dropout=args.use_dropout,
                         g_n_downsampling=args.g_n_downsampling,
                         dropout=args.dropout)
    model.to(device)
    loss_record = []

    for epoch in range(args.start_epochs, args.start_epochs +
                       args.train_epochs):
        Logger("epoch: ", epoch + 1)
        Logger(time.asctime())
        loss = run_one_epoch(model, trainloader,
                             training=True, max_length=args.max_length)
        Logger(loss)
        loss_record.append(loss)
        if (epoch + 1) % args.save_freq == 0:
            torch.save(model, os.path.join(args.save_path, args.save_prefix +
                                           '_epoch_' + str(epoch + 1) + '.model'))
        if (epoch + 1) % args.img_freq == 0:
            model.save_current_batch(args.save_path, str(epoch + 1))
    Logger("done")
    Logger(time.asctime())
    Logger(loss_record)
    # torch.save(model, os.path.join(args.save_path, args.save_prefix +
    #                                '_epoch_' + str(epoch + 1) + '.model'))
