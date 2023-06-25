import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

#生成模态v
class Generator_v(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(Generator_v, self).__init__()
        self.input_nc = input_nc  # input_nc
        self.output_nc = output_nc

        model = [nn.Linear(self.input_nc, 2048),
                 nn.ReLU(True),
                 nn.Linear(2048, self.output_nc),
                 nn.ReLU()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        G_prob = self.model(input)
        return G_prob

#生成模态a
class Generator_a(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(Generator_a, self).__init__()
        self.input_nc = input_nc  # input_nc
        self.output_nc = output_nc

        model = [nn.Linear(self.input_nc, 2048),
                 nn.ReLU(True),
                 nn.Linear(2048, self.output_nc),
                 nn.ReLU()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        G_prob = self.model(input)
        return G_prob

#生成模态t
class Generator_t(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(Generator_t, self).__init__()
        self.input_nc = input_nc  # input_nc
        self.output_nc = output_nc

        model = [nn.Linear(self.input_nc, 2048),
                 nn.ReLU(True),
                 nn.Linear(2048, self.output_nc),
                 nn.ReLU()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        G_prob = self.model(input)
        return G_prob
    #(23,2048)


class Discriminator_vat(nn.Module):
    def __init__(self, input_nc, n_layers=3):
        super(Discriminator_vat, self).__init__()
        self.input_nc = input_nc
        sequence = [
            nn.Linear(self.input_nc, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1)
        ]

        self.model = nn.Sequential(*sequence)


    def forward(self, input):

        D_logit = self.model(input)
        D_prob = F.sigmoid(D_logit)

        return D_logit, D_prob

class G_D_loss(nn.Module):
    def __init__(self, input_size, out_size):
        super(G_D_loss, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.G_V = Generator_v(self.input_size, self.out_size)
        self.G_A = Generator_a(self.input_size, self.out_size)
        self.G_T = Generator_t(self.input_size, self.out_size)
        self.D_VAT = Discriminator_vat(self.input_size)

    def forward(self,input_vv, input_va, input_vt, input_vat):

        g_vat_v = self.G_V(input_vat)
        g_vat_t = self.G_T(input_vat)
        g_vat_a = self.G_A(input_vat)

        D_logit_real, D_real = self.D_VAT(input_vat)
        D_logit_fake_1, D_fake_1 = self.D_VAT(input_vv)
        D_logit_fake_2, D_fake_2 = self.D_VAT(input_va)
        D_logit_fake_3, D_fake_3 = self.D_VAT(input_vt)

        loss_D_1= - torch.mean(torch.log(D_real) + torch.log(1. - D_fake_1))
        loss_D_2 = - torch.mean(torch.log(D_real) + torch.log(1. - D_fake_2))
        loss_D_3 = - torch.mean(torch.log(D_real) + torch.log(1. - D_fake_3))
        loss_D = loss_D_1 + loss_D_2 + loss_D_3

        return g_vat_v, g_vat_a, g_vat_t, loss_D

        # G_loss = torch.mean(torch.log(1. - prob_artist1))

if __name__ == '__main__':
    D = torch.rand(23, 2048)
    B = torch.rand(23, 2048)
    C = torch.rand(23, 2048)

    # net = Generator(2048, 2048)
    net = Discriminator_vat(2048)

    output = net(D)
    print(output)
    print('over')