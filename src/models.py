
import torch
from torch import nn
from src.multihead import MultiHeadAttention,MultiHeadAttention_vat
from src.GAN import G_D_loss
import torch.nn.functional as F
from scipy import io as scio
import os
from torch.utils.data import DataLoader

class PreNet(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, dropout):

        super(PreNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, out_size)


    def forward(self, data):
        """
        Args:
            data:  tensor of shape (batch_size, in_size)
        """
        normed = self.norm(data)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3

class MULTModel(nn.Module):

    def __init__(self, config):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.config = config
        self.embed_size = int(config['embed_size'])
        self.v = float(self.config['v'])
        self.a = float(self.config['a'])
        self.t = float(self.config['t'])
        self.D = float(self.config['D'])
        self.hid_dim = int(config['hid_dim'])
        self.attn_dropout_t = float(self.config['attn_dropout'])
        self.attn_dropout_a = float(self.config['attn_dropout_a'])
        self.attn_dropout_v = float(self.config['attn_dropout_v'])
        self.attn_dropout_vat = float(self.config['attn_dropout_vat'])
        self.attn_dropout = float(self.config['attn_dropout'])
        self.out_dim = int(config['out_dim'])
        self.output_dim = int(config['output_dim'])
        self.d_k, self.d_v = [int(x) for x in self.config['d_i'].split(',')]
        self.n_heads = int(config['n_heads'])
        self.dropout = float(self.config['dropout'])
        self.pre_layer = PreNet(self.embed_size, self.hid_dim, self.out_dim, self.dropout)
        self.out_layer = nn.Linear(self.out_dim, self.output_dim)
        self.gan_layer = G_D_loss(self.embed_size, self.embed_size)
        self.criterionCycle = torch.nn.L1Loss()


        # 2. Crossmodal Attentions

        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_v_with_t = self.get_network(self_type='vt')
        self.trans_v_with_v = self.get_network(self_type='vv')
        self.trans_v_with_at = self.encoder_vat(self_type='vat')
        #self-attention
        # self.trans_a_mem = self.get_network(self_type='a_mem')
        # self.trans_v_mem = self.get_network(self_type='v_mem')
        # self.trans_t_mem = self.get_network(self_type='t_mem')


    def get_network(self, self_type='v'):
        if self_type in ['va']:
            attn_dropout = self.attn_dropout_a

        elif self_type in ['vt']:
            attn_dropout = self.attn_dropout_t

        elif self_type in ['vv']:
            attn_dropout = self.attn_dropout_v

        return MultiHeadAttention(embed_size=self.embed_size,
                                  attn_dropout=attn_dropout,
                                  d_k=self.d_k,
                                  d_v=self.d_v,
                                  n_heads=self.n_heads)

    def encoder_vat(self,self_type='vat'):
         if self_type in ['vat']:
             attn_dropout = self.attn_dropout_vat

         return MultiHeadAttention_vat(embed_size=self.embed_size,
                                   attn_dropout=attn_dropout,
                                   d_k=self.d_k,
                                   d_v=self.d_v,
                                   n_heads=self.n_heads)


        # elif self_type == 't_mem':
        #     attn_dropout = self.attn_dropout
        # elif self_type == 'a_mem':
        #     attn_dropout = self.attn_dropout
        # elif self_type == 'v_mem':
        #     attn_dropout = self.attn_dropout


    def forward(self, x_v, x_a, x_t):
        """
         audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #
        x_a = torch.tensor(x_a)
        x_a = x_a.unsqueeze(dim=1)
        # x_a = x_a.transpose(1, 2)
        #(23,1,2048)


        x_v = torch.tensor(x_v)
        x_v = x_v.unsqueeze(dim=1)
        # x_v = x_v.transpose(1, 2)

        x_t = torch.tensor(x_t)
        x_t = x_t.unsqueeze(dim=1)
        # x_t = x_t.transpose(1, 2)


        # Project the tract/visual/audio features
        #
        # x_a = x_a.permute(2, 0, 1)
        # x_v = x_v.permute(2, 0, 1)
        # x_t = x_t.permute(2, 0, 1)

        h_v_with_ts = self.trans_v_with_t(x_t, x_v, x_v)
        h_v_with_as = self.trans_v_with_a(x_a, x_v, x_v)
        h_v_with_vs = self.trans_v_with_v(x_v, x_v, x_v)
        h_v_with_ats = self.trans_v_with_at(x_a, x_v, x_v, x_t, x_v, x_v)

        # h_v_with_ats = torch.cat([self.trans_v_with_at(x_a, x_v, x_v), self.trans_v_with_at(x_t, x_v, x_v)], dim=2)
        # h_v_with_ats = self.vat(h_v_with_ats)

        h_v_with_ts = h_v_with_ts.permute(1, 0, 2)
        h_v_with_as = h_v_with_as.permute(1, 0, 2)
        h_v_with_vs = h_v_with_vs.permute(1, 0, 2)
        h_v_with_ats = h_v_with_ats.permute(1, 0, 2)
        # (1,23,2048)

        last_h_v = h_v_with_vs[0]
        last_h_t = h_v_with_ts[0]
        last_h_a = h_v_with_as[0]
        last_h_vat = h_v_with_ats[0]

        g_vat_v, g_vat_a, g_vat_t, loss_D = self.gan_layer(last_h_v, last_h_a, last_h_t, last_h_vat)
        # (23,2048)

        g_vat_v = g_vat_v.unsqueeze(dim=1)
        g_vat_t = g_vat_t.unsqueeze(dim=1)
        g_vat_a = g_vat_a.unsqueeze(dim=1)

        # (23,1,2048)

        # zero = torch.zeros_like(g_vat_v, requires_grad=False)
        output_vv = (self.trans_v_with_v(g_vat_v, g_vat_v, g_vat_v)).permute(1, 0, 2)
        output_va = (self.trans_v_with_a(g_vat_a, g_vat_v, g_vat_v)).permute(1, 0, 2)
        output_vt = (self.trans_v_with_t(g_vat_t, g_vat_v, g_vat_v)).permute(1, 0, 2)
        #(1,23,2048)

        output_vv = output_vv[0]
        output_va = output_va[0]
        output_vt = output_vt[0]
        #(23,2048)

        loss_G_cycle = self.criterionCycle(output_vv, last_h_vat)*self.v+self.criterionCycle(output_va, last_h_vat)*self.a+self.criterionCycle(output_vt, last_h_vat)*self.t
        loss_total = self.D*loss_D + loss_G_cycle




        # last_hs = torch.cat([last_h_v, last_h_t, last_h_a], dim=1)
        # last_hs = last_h_v
        #(23,6144)

        #
        output = self.pre_layer(last_h_vat)
        # output = self.pre_layer(last_h_a)
        output = self.out_layer(output)


        # loss_lrr = (loss_vv).mean()
        # h_t = self.trans_t_mem(h_v_with_ts)
        # h_a = self.trans_a_mem(h_v_with_as)
        # h_v = self.trans_v_mem(h_v_with_vs)

        # A residual block


        #(23,63)

        return output, loss_total

if __name__ == '__main__':
    D = torch.randn(23, 2048)
    B = torch.randn(23, 2048)
    C = torch.randn(23, 2048)
    config = dict()
    config['embed_size'] = '2048'
    config['v'] = '0.5'
    config['a'] = '0.5'
    config['t'] = '0.5'
    config['D'] = '0.5'
    config['modal_num'] = '3'
    config['d_i'] = '512, 512'
    config['hid_dim'] = '512'
    config['out_dim'] = '128'
    config['n_heads'] = '4'
    config['output_dim'] = '63'
    config['attn_dropout'] = '0.1'
    config['attn_dropout_a'] = '0.1'
    config['attn_dropout_t'] = '0.1'
    config['attn_dropout_v'] = '0.1'
    config['attn_dropout_vat'] = '0.1'
    config['dropout'] = '0.0'
    config['rank'] = '4'
    net = MULTModel(config=config)
    output = net(D, B, C)
    print(output)

if __name__ == '__main__':
    D = torch.randn(23, 2048)
    B = torch.randn(23, 2048)
    C = torch.randn(23, 2048)
    config = dict()
    config['embed_size'] = '2048'
    config['v'] = '0.5'
    config['a'] = '0.5'
    config['t'] = '0.5'
    config['D'] = '0.5'
    config['modal_num'] = '3'
    config['d_i'] = '512, 512'
    config['hid_dim'] = '512'
    config['out_dim'] = '128'
    config['n_heads'] = '4'
    config['output_dim'] = '63'
    config['attn_dropout'] = '0.1'
    config['attn_dropout_a'] = '0.1'
    config['attn_dropout_t'] = '0.1'
    config['attn_dropout_v'] = '0.1'
    config['attn_dropout_vat'] = '0.1'
    config['dropout'] = '0.0'
    config['rank'] = '4'
    net = MULTModel(config=config)
    output = net(D, B, C)
    print(output)


