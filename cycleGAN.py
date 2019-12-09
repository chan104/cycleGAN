import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import os.path

import utils
from base_net import Generator, Discriminator
from adam import Adam


class cycleGAN(nn.Module):
    def __init__(self, A_channels, B_channels, device, lr=0.0002, beta1=0.5,
                 lambda_A=10, lambda_B=10, lambda_idt=0.5, scheduler_gamma=0.3,
                 scheduler_step_size=20, ngf=64, ndf=64, g_blocks=9, n_layers=3,
                 use_dropout=False, g_n_downsampling=2, dropout=0.5,
                 scheduler_type='StepLR'):
        super(cycleGAN, self).__init__()
        self.netG_A2B = Generator(A_channels, B_channels, ngf=ngf, n_blocks=g_blocks,
                                  use_dropout=use_dropout,
                                  n_downsampling=g_n_downsampling, dropout=dropout)
        self.netG_B2A = Generator(B_channels, A_channels, ngf=ngf, n_blocks=g_blocks,
                                  use_dropout=use_dropout,
                                  n_downsampling=g_n_downsampling, dropout=dropout)
        self.netD_A = Discriminator(A_channels, ndf=ndf, n_layers=n_layers,
                                    use_dropout=use_dropout, dropout=dropout)
        self.netD_B = Discriminator(B_channels, ndf=ndf, n_layers=n_layers,
                                    use_dropout=use_dropout, dropout=dropout)

        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_idt = lambda_idt

        self.device = device

        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        self.reset_optimizer(lr=lr, beta1=beta1, scheduler_gamma=scheduler_gamma,
                             scheduler_step_size=scheduler_step_size,
                             scheduler_type=scheduler_type)
        self.mode = "both_sides"

    def reset_optimizer(self, lr=0.0002, beta1=0.5, scheduler_gamma=0.3,
                        scheduler_step_size=5, scheduler_type='StepLR'):
        self.optimizer_G = Adam(list(self.netG_A2B.parameters()) +
                                list(self.netG_B2A.parameters()),
                                lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = Adam(list(self.netD_A.parameters()) +
                                list(self.netD_B.parameters()),
                                lr=lr, betas=(beta1, 0.999))

        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        if scheduler_type == 'StepLR':
            self.schedulers = [lr_scheduler.StepLR(
                optimizer, scheduler_step_size, gamma=scheduler_gamma)
                for optimizer in self.optimizers]
        elif scheduler_type == 'linear_decay':
            self.schedulers = [lr_scheduler.LambdaLR(
                optimizer, utils.Linear_decay(scheduler_step_size))
                for optimizer in self.optimizers]
        else:
            raise Exception('unknown scheduler type')

    def criterionGAN(self, out, label):
        if label:
            loss = torch.mean((out - 1)**2)
        else:
            loss = torch.mean(out**2)
        return loss

    def calc_generator_loss(self):
        self.loss_G_A2B = self.criterionGAN(self.netD_B(self.fake_B), True)
        self.loss_G_B2A = self.criterionGAN(self.netD_A(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(
            self.rec_A, self.real_A) * self.lambda_A
        self.loss_cycle_B = self.criterionCycle(
            self.rec_B, self.real_B) * self.lambda_B

        if self.lambda_idt > 0:
            self.idt_B = self.netG_A2B(self.real_B)
            self.loss_idt_B = self.criterionIdt(
                self.idt_B, self.real_B) * self.lambda_B * self.lambda_idt
            self.idt_A = self.netG_B2A(self.real_A)
            self.loss_idt_A = self.criterionIdt(
                self.idt_A, self.real_A) * self.lambda_A * self.lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G = self.loss_G_A2B + self.loss_G_B2A + self.loss_cycle_A + \
            self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

    def get_discriminator_loss(self, net, real, fake):
        loss_D_real = self.criterionGAN(net(real), True)
        loss_D_fake = self.criterionGAN(net(fake.detach()), False)
        return (loss_D_real + loss_D_fake) * 0.5

    def calc_discriminator_loss(self):
        self.loss_D_A = self.get_discriminator_loss(
            self.netD_A, self.real_A, self.fake_A)
        self.loss_D_B = self.get_discriminator_loss(
            self.netD_B, self.real_B, self.fake_B)
        self.loss_D = self.loss_D_A + self.loss_D_B

    def process_one_input(self, x, training=True):
        self.real_A = x['A'].to(self.device)
        self.real_B = x['B'].to(self.device)
        self.fake_B = self.netG_A2B(self.real_A)
        self.rec_A = self.netG_B2A(self.fake_B)
        self.fake_A = self.netG_B2A(self.real_B)
        self.rec_B = self.netG_A2B(self.fake_A)

        if not training:
            self.calc_generator_loss()
            self.calc_discriminator_loss()
            return

        for p in self.netD_A.parameters():
            p.requires_grad_(False)
        for p in self.netD_B.parameters():
            p.requires_grad_(False)

        self.optimizer_G.zero_grad()
        self.calc_generator_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

        for p in self.netD_A.parameters():
            p.requires_grad_(True)
        for p in self.netD_B.parameters():
            p.requires_grad_(True)

        self.optimizer_D.zero_grad()
        self.calc_discriminator_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

    def schedulers_step(self):
        for s in self.schedulers:
            s.step()

    def get_loss(self):
        ret = [self.loss_G.item(), self.loss_D_A.item(), self.loss_D_B.item()]
        ret += [self.loss_G_A2B.item(), self.loss_G_B2A.item(),
                self.loss_cycle_A.item(), self.loss_cycle_B.item(),
                float(self.loss_idt_A), float(self.loss_idt_B)]
        return ret

    def save_current_batch(self, path, prefix, start_id=0):
        if self.mode == "both_sides":
            img_names = ['real_A', 'real_B',
                         'fake_A', 'fake_B', 'rec_A', 'rec_B']
            if (self.lambda_idt > 0):
                img_names += ['idt_A', 'idt_B']
        elif self.mode == "A2B":
            img_names = ['real_A', 'fake_B', 'rec_A']
        elif self.mode == "B2A":
            img_names = ['real_B', 'fake_A', 'rec_B']
        for name in img_names:
            imgs = getattr(self, name)
            for i in range(imgs.shape[0]):
                img = imgs[i].detach().cpu().numpy()
                utils.savefig(img, os.path.join(
                    path, prefix + '_' + str(start_id + i) + '_' + name))

    def one_side_process(self, x):
        if self.mode == "A2B":
            self.real_A = x.to(self.device)
            self.fake_B = self.netG_A2B(self.real_A)
            self.rec_A = self.netG_B2A(self.fake_B)
        elif self.mode == "B2A":
            self.real_B = x.to(self.device)
            self.fake_A = self.netG_B2A(self.real_B)
            self.rec_B = self.netG_A2B(self.fake_A)
        else:
            print("wrong mode")
