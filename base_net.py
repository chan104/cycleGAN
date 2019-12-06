import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, channels, use_dropout=True, dropout=0.5):
        super(BasicBlock, self).__init__()
        model = [nn.Conv2d(channels, channels, 3, padding=1),
                 nn.BatchNorm2d(channels),
                 nn.ReLU(inplace=True)]

        if use_dropout:
            model += [nn.Dropout(dropout)]

        model += [nn.Conv2d(channels, channels, 3, padding=1),
                  nn.BatchNorm2d(channels)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        temp = self.model(x) + x
        return temp


class Generator(nn.Module):
    def __init__(self, input_channel, output_channel, ngf=64, n_blocks=9,
                 use_dropout=True, n_downsampling=2, dropout=0.5):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channel, ngf, kernel_size=7, padding=0),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU(inplace=True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      nn.BatchNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [BasicBlock(ngf * mult,
                                 use_dropout=use_dropout, dropout=0.5)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult // 2,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      nn.BatchNorm2d(ngf * mult // 2),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_channel, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, input_c, ndf=64, n_downsampling=2, width=4, padw=1,
                 n_blocks=4, use_dropout=True, dropout=0.5):
        super(Discriminator, self).__init__()
        model = []
        current_c = input_c
        for n in range(n_downsampling):
            model += [nn.Conv2d(current_c, ndf * 2 ** n,
                                kernel_size=width, stride=2, padding=padw),
                      nn.BatchNorm2d(ndf * 2 ** n),
                      nn.LeakyReLU(0.2, True)]
            current_c = ndf * 2 ** n

        for i in range(n_blocks):
            model += [BasicBlock(current_c,
                                 use_dropout=use_dropout, dropout=0.5)]

        model += [nn.Conv2d(current_c, 1,
                            kernel_size=width, stride=1, padding=padw)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        temp = self.model(input)

        return temp
