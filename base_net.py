import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_dropout=True, use_bias=True, dropout=0.5):
        super(ResidualBlock, self).__init__()
        model = [nn.Conv2d(channels, channels, 3,
                           padding=1, bias=use_bias),
                 nn.BatchNorm2d(channels),
                 nn.ReLU(inplace=True)]

        if use_dropout:
            model += [nn.Dropout(dropout)]

        model += [nn.Conv2d(channels, channels, 3,
                            padding=1, bias=use_bias),
                  nn.BatchNorm2d(channels)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        temp = self.model(x) + x
        return temp


class Generator(nn.Module):
    def __init__(self, input_channel, output_channel, ngf=64, n_blocks=9,
                 use_bias=True, use_dropout=True, n_downsampling=2):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channel, ngf, kernel_size=7,
                           padding=0, bias=use_bias),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      nn.BatchNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResidualBlock(ngf * mult, use_dropout=use_dropout,
                                    use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      nn.BatchNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_channel, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class Encoder(nn.Module):
    def __init__(self, input_channel, ngf=64, n_blocks=5,
                 use_bias=True, use_dropout=True, n_downsampling=2):
        super(Encoder, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channel, ngf, kernel_size=7,
                           padding=0, bias=use_bias),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      nn.BatchNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResidualBlock(ngf * mult, use_dropout=use_dropout,
                                    use_bias=use_bias)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class Decoder(nn.Module):
    def __init__(self, output_channel, ngf=64, n_blocks=5,
                 use_bias=True, use_dropout=True, n_downsampling=2):
        super(Decoder, self).__init__()
        model = []

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResidualBlock(ngf * mult, use_dropout=use_dropout,
                                    use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      nn.BatchNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_channel, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
