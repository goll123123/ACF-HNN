import torch
from torch import nn
from models import HGNN_conv, FALayer
import torch.nn.functional as F


class HGNNConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X, hg):
        X = self.theta(X)
        print(X.shape)
        X = hg.smoothing_with_HGNN(X)
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X



class HGNN_dhg(nn.Module):
    r"""The HGNN model proposed in `Hypergraph Neural Networks <https://arxiv.org/pdf/1809.09401>`_ paper (AAAI 2019).

    Args:
        ``in_channels`` (``int``): :math:`C_{in}` is the number of input channels.
        ``hid_channels`` (``int``): :math:`C_{hid}` is the number of hidden channels.
        ``num_classes`` (``int``): The Number of class of the classification task.
        ``use_bn`` (``bool``): If set to ``True``, use batch normalization. Defaults to ``False``.
        ``drop_rate`` (``float``, optional): Dropout ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X, hg):
        r"""The forward function.

        Args:
            ``X`` (``torch.Tensor``): Input vertex feature matrix. Size :math:`(N, C_{in})`.
            ``hg`` (``dhg.Hypergraph``): The hypergraph structure that contains :math:`N` vertices.
        """
        for layer in self.layers:
            X = layer(X, hg)
        return X


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, nlayers, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)
        # if nlayers == 1:
        #     hgc = HGNN_conv(in_ch, n_class)
        # else:
        #     self.hgc = nn.ModuleList()
        #     self.hgc.append(HGNN_conv(in_ch, n_hid))
        #     for _ in range(nlayers-2):
        #         self.hgc.append(HGNN_conv(n_hid, n_hid))
        #     self.hgc.append(HGNN_conv(n_hid, n_class))

    # def forward(self, x, G, V, E):
    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        # if self.nlayers == 1:
        #     x = self.hgc(x, G, V, E)
        # else:
        #     for i in range(self.nlayers-1):
        #         x = F.relu(self.hgc[i](x,G,V,E))
        #         x = F.dropout(x, self.dropout)
        #     x = self.hgc[-1](x,G,V,E)
        
        return F.log_softmax(x)


class FAHGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, nlayers, dropout=0.5):
        super(FAHGNN, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        # self.hgc1 = HGNN_conv(in_ch, n_hid)
        # self.hgc2 = HGNN_conv(n_hid, n_class)
        if nlayers == 1:
            hgc = FALayer(in_ch, n_class, dropout)
        else:
            self.hgc = nn.ModuleList()
            self.hgc.append(FALayer(in_ch, n_hid, dropout))
            for _ in range(nlayers - 2):
                self.hgc.append(FALayer(n_hid, n_hid, dropout))
            self.hgc.append(FALayer(n_hid, n_class, dropout))

    def forward(self, x, G):
        # x = F.relu(self.hgc1(x, G))
        # x = F.dropout(x, self.dropout)
        # x = self.hgc2(x, G)
        if self.nlayers == 1:
            x = self.hgc(x, G)
        else:
            for i in range(self.nlayers - 1):
                x = F.relu(self.hgc[i](x, G))
                x = F.dropout(x, self.dropout)
            x = self.hgc[-1](x, G)

        return x