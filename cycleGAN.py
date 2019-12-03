import torch
import torch.nn as nn
import itertools
from torch.optim import lr_scheduler
import utils

from base_net import Generator, Discriminator
from adam import Adam
import os.path


class cycleGAN(nn.Module):
    def __init__(self, A_channels, B_channels, device, lr=0.0002, beta1=0.5,
                 lambda_A=10, lambda_B=10, lambda_idt=5, scheduler_gamma=0.3,
                 scheduler_step_size=5):
        super(cycleGAN, self).__init__()
        self.netG_A2B = Generator(A_channels, B_channels)
        self.netG_B2A = Generator(B_channels, A_channels)
        self.netD_A = Discriminator(A_channels)
        self.netD_B = Discriminator(B_channels)

        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_idt = lambda_idt

        self.device = device

        self.optimizer_G = Adam(itertools.chain(
            self.netG_A2B.parameters(), self.netG_B2A.parameters()),
            lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = Adam(itertools.chain(
            self.netD_A.parameters(), self.netD_B.parameters()),
            lr=lr, betas=(beta1, 0.999))

        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.schedulers = [lr_scheduler.StepLR(
            optimizer, scheduler_step_size, gamma=scheduler_gamma)
            for optimizer in self.optimizers]

        self.img_names = ['real_A', 'real_B',
                          'fake_A', 'fake_B', 'rec_A', 'rec_B']
        if (self.lambda_idt > 0):
            self.img_names += ['idt_A', 'idt_B']

    def reset_optimizer(self, lr=0.0002, beta1=0.5, scheduler_gamma=0.3,
                        scheduler_step_size=5):
        self.optimizer_G = Adam(itertools.chain(
            self.netG_A2B.parameters(), self.netG_B2A.parameters()),
            lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = Adam(itertools.chain(
            self.netD_A.parameters(), self.netD_B.parameters()),
            lr=lr, betas=(beta1, 0.999))

        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        self.schedulers = [lr_scheduler.StepLR(
            optimizer, scheduler_step_size, gamma=scheduler_gamma)
            for optimizer in self.optimizers]

    def save_current_batch(self, path, prefix):
        for name in self.img_names:
            if isinstance(name, str):
                imgs = getattr(self, name)
                for i in range(imgs.shape[0]):
                    img = imgs[0].detach().cpu().numpy()
                    utils.savefig(img, os.path.join(
                        path, prefix + '_' + str(i) + '_' + name))

    def forward(self):
        self.fake_B = self.netG_A2B(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B2A(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B2A(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A2B(self.fake_A)   # G_A(G_B(B))

    def criterionGAN(self, out, label):
        loss = 0
        if label:
            loss = torch.mean((out - 1)**2)
        else:
            loss = torch.mean(out**2)
        return loss

    def calc_generator_loss(self):
        if self.lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A2B(self.real_B)
            self.loss_idt_A = self.criterionIdt(
                self.idt_A, self.real_B) * self.lambda_B * self.lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B2A(self.real_A)
            self.loss_idt_B = self.criterionIdt(
                self.idt_B, self.real_A) * self.lambda_A * self.lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A2B = self.criterionGAN(self.netD_B(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B2A = self.criterionGAN(self.netD_A(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(
            self.rec_A, self.real_A) * self.lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(
            self.rec_B, self.real_B) * self.lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A2B + self.loss_G_B2A + self.loss_cycle_A + \
            self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B

    def get_discriminator_loss(self, net, real, fake):
        pred_real = net(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = net(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def calc_discriminator_loss(self):
        self.loss_D_A = self.get_discriminator_loss(
            self.netD_A, self.real_A, self.fake_A)
        self.loss_D_B = self.get_discriminator_loss(
            self.netD_B, self.real_B, self.fake_B)

    def process_one_input(self, x, training=True):
        self.real_A = x['A'].to(self.device)
        self.real_B = x['B'].to(self.device)
        self.forward()

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
        self.loss_D_A.backward()
        self.loss_D_B.backward()
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
