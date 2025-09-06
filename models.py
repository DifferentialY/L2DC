import torch.nn as nn
import torch
import math
from torch.nn.functional import normalize
import torch.nn.init as init
import numpy as np


class MLP(nn.Module):

    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=False):
        super(MLP, self).__init__()
        self.input_dims = input_dims  # 输入层
        self.hid_dims = hid_dims  # 隐藏层
        self.output_dims = out_dims  # 输出层
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(self.input_dims, self.hid_dims[0]))
        self.layers.append(nn.ReLU())
        for i in range(len(hid_dims) - 1):
            self.layers.append(nn.Linear(self.hid_dims[i], self.hid_dims[i + 1]))
            self.layers.append(nn.ReLU())

        self.out_layer = nn.Linear(self.hid_dims[-1], self.output_dims)
        if kaiming_init:
            self.reset_parameters()

    # 初始化
    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight)
                init.zeros_(layer.bias)
        init.xavier_uniform_(self.out_layer.weight)
        init.zeros_(self.out_layer.bias)

    #
    def forward(self, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        h = self.out_layer(h)
        h = torch.tanh_(h)
        return h


class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))

    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)


class SENet(nn.Module):

    def __init__(self, input_dims, hid_dims, out_dims, kaiming_init=True):
        super(SENet, self).__init__()
        self.input_dims = input_dims
        self.hid_dims = hid_dims
        self.out_dims = out_dims
        self.kaiming_init = kaiming_init
        self.shrink = 1.0 / out_dims

        self.net_q = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.net_k = MLP(input_dims=self.input_dims,
                         hid_dims=self.hid_dims,
                         out_dims=self.out_dims,
                         kaiming_init=self.kaiming_init)

        self.thres = AdaptiveSoftThreshold(1)

    def query_embedding(self, queries):
        q_emb = self.net_q(queries)
        return q_emb

    def key_embedding(self, keys):
        k_emb = self.net_k(keys)
        return k_emb

    def get_coeff(self, q_emb, k_emb):
        c = self.thres(q_emb.mm(k_emb.t()))
        return self.shrink * c

    def forward(self, queries, keys):
        q = self.query_embedding(queries)
        k = self.key_embedding(keys)
        out = self.get_coeff(q_emb=q, k_emb=k)
        return out


def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()


class SUREfcNoisyMNIST(nn.Module):
    def __init__(self):
        super(SUREfcNoisyMNIST, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(10, 1024),
            # nn.BatchNorm1d(1024),
            nn.Linear(1024, 10),
            nn.Softmax(dim=1)
        )

        self.decoder0 = nn.Sequential(nn.Linear(10, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 784))
        self.decoder1 = nn.Sequential(nn.Linear(10, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 784))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        z0 = self.decoder0(h0)
        z1 = self.decoder1(h1)

        return h0, h1, pseudo0, pseudo1, z0, z1

    def forward_plot(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        return h0, h1

    def forward_cluster(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        pred0 = torch.argmax(pseudo0, dim=1)
        pred1 = torch.argmax(pseudo1, dim=1)
        return pseudo0, pseudo1, pred0, pred1

    def forward_pseudo(self, h0, h1):
        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)
        return pseudo0, pseudo1

    def forward_generative(self, h, view):
        if view == 0:
            z = self.decoder0(h)
        elif view == 1:
            z = self.decoder1(h)
        return z


class SUREfcCaltech(nn.Module):
    def __init__(self):
        super(SUREfcCaltech, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(1984, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(10, 2048),
            # nn.BatchNorm1d(1024),
            nn.Linear(2048, 102),
            nn.Softmax(dim=1)
        )

        self.decoder0 = nn.Sequential(nn.Linear(10, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1984))
        self.decoder1 = nn.Sequential(nn.Linear(10, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 512))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        z0 = self.decoder0(h0)
        z1 = self.decoder1(h1)

        return h0, h1, pseudo0, pseudo1, z0, z1

    def forward_plot(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        return h0, h1

    def forward_cluster(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        pred0 = torch.argmax(pseudo0, dim=1)
        pred1 = torch.argmax(pseudo1, dim=1)
        return pseudo0, pseudo1, pred0, pred1

    def forward_pseudo(self, h0, h1):
        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)
        return pseudo0, pseudo1

    def forward_generative(self, h, view):
        if view == 0:
            z = self.decoder0(h)
        elif view == 1:
            z = self.decoder1(h)
        return z


class SUREfcScene(nn.Module):  # 20, 59
    def __init__(self):
        super(SUREfcScene, self).__init__()

        self.encoder0 = nn.Sequential(
            nn.Linear(20, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(59, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
            # Varying the number of layers of W can obtain the representations with different shapes.
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(10, 1024),
            # nn.BatchNorm1d(1024),
            nn.Linear(1024, 15),
            nn.Softmax(dim=1)
        )

        self.decoder0 = nn.Sequential(nn.Linear(10, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 20))
        self.decoder1 = nn.Sequential(nn.Linear(10, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 1024),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(1024, 59))

    # self-expressive
    def get_c(self, x0, x1):
        union = torch.cat([x0, x1], 1)  # n,2d
        q_union = self.attention.query_embedding(union)  # n,2d
        k_union = self.attention.key_embedding(union)  # n,2d
        c_union = self.attention(q_union, k_union)  # n,n
        c_union = c_union - torch.diag_embed(torch.diag(c_union))
        return c_union


    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        z0 = self.decoder0(h0)
        z1 = self.decoder1(h1)

        return h0, h1, pseudo0, pseudo1, z0, z1

    def forward_plot(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        return h0, h1

    def forward_cluster(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        pred0 = torch.argmax(pseudo0, dim=1)
        pred1 = torch.argmax(pseudo1, dim=1)
        return pseudo0, pseudo1, pred0, pred1

    def forward_pseudo(self, h0, h1):
        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)
        return pseudo0, pseudo1

    def forward_generative(self, h, view):
        if view == 0:
            z = self.decoder0(h)
        elif view == 1:
            z = self.decoder1(h)
        return z


class SUREfcReuters(nn.Module):
    def __init__(self):
        super(SUREfcReuters, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(10, 10),
            # Varying the number of layers of W can obtain the representations with different shapes.
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(10, 1024),
            nn.Linear(1024, 5),
            nn.Softmax(dim=1)
        )

        self.decoder0 = nn.Sequential(nn.Linear(10, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 10))
        self.decoder1 = nn.Sequential(nn.Linear(10, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 10))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        z0 = self.decoder0(h0)
        z1 = self.decoder1(h1)

        return h0, h1, pseudo0, pseudo1, z0, z1

    def forward_plot(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        return h0, h1

    def forward_cluster(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        pred0 = torch.argmax(pseudo0, dim=1)
        pred1 = torch.argmax(pseudo1, dim=1)
        return pseudo0, pseudo1, pred0, pred1

    def forward_pseudo(self, h0, h1):
        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)
        return pseudo0, pseudo1

    def forward_generative(self, h, view):
        if view == 0:
            z = self.decoder0(h)
        elif view == 1:
            z = self.decoder1(h)
        return z


class SUREfcMNISTUSPS(nn.Module):
    def __init__(self):
        super(SUREfcMNISTUSPS, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(10, 1024),
            # nn.BatchNorm1d(1024),
            nn.Linear(1024, 10),
            nn.Softmax(dim=1)
        )

        self.decoder0 = nn.Sequential(nn.Linear(10, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 784))
        self.decoder1 = nn.Sequential(nn.Linear(10, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 256))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        z0 = self.decoder0(h0)
        z1 = self.decoder1(h1)

        return h0, h1, pseudo0, pseudo1, z0, z1

    def forward_plot(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        return h0, h1

    def forward_cluster(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        pred0 = torch.argmax(pseudo0, dim=1)
        pred1 = torch.argmax(pseudo1, dim=1)
        return pseudo0, pseudo1, pred0, pred1

    def forward_pseudo(self, h0, h1):
        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)
        return pseudo0, pseudo1

    def forward_generative(self, h, view):
        if view == 0:
            z = self.decoder0(h)
        elif view == 1:
            z = self.decoder1(h)
        return z


class SUREfcDeepCaltech(nn.Module):
    def __init__(self):
        super(SUREfcDeepCaltech, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True)
        )

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(32, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(1024),
            # nn.Dropout(1024),
            nn.Linear(1024, 101),
            nn.Softmax(dim=1)
        )

        self.decoder0 = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 4096))
        self.decoder1 = nn.Sequential(nn.Linear(32, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 4096))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        z0 = self.decoder0(h0)
        z1 = self.decoder1(h1)

        return h0, h1, pseudo0, pseudo1, z0, z1

    def forward_plot(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        return h0, h1

    def forward_cluster(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        pred0 = torch.argmax(pseudo0, dim=1)
        pred1 = torch.argmax(pseudo1, dim=1)
        return pseudo0, pseudo1, pred0, pred1

    def forward_pseudo(self, h0, h1):
        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)
        return pseudo0, pseudo1

    def forward_generative(self, h, view):
        if view == 0:
            z = self.decoder0(h)
        elif view == 1:
            z = self.decoder1(h)
        return z


class SUREfcDeepAnimal(nn.Module):
    def __init__(self):
        super(SUREfcDeepAnimal, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(True)
        )

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(10, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(1024),
            # nn.Dropout(1024),
            nn.Linear(1024, 50),
            nn.Softmax(dim=1)
        )

        self.decoder0 = nn.Sequential(nn.Linear(10, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 4096))
        self.decoder1 = nn.Sequential(nn.Linear(10, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
                                      nn.Linear(1024, 4096))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        z0 = self.decoder0(h0)
        z1 = self.decoder1(h1)

        return h0, h1, pseudo0, pseudo1, z0, z1

    def forward_plot(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        return h0, h1

    def forward_cluster(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))

        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)

        pred0 = torch.argmax(pseudo0, dim=1)
        pred1 = torch.argmax(pseudo1, dim=1)
        return pseudo0, pseudo1, pred0, pred1

    def forward_pseudo(self, h0, h1):
        pseudo0 = self.label_contrastive_module(h0)
        pseudo1 = self.label_contrastive_module(h1)
        return pseudo0, pseudo1

    def forward_generative(self, h, view):
        if view == 0:
            z = self.decoder0(h)
        elif view == 1:
            z = self.decoder1(h)
        return z

