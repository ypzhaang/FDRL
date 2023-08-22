# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Nets.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import json
import numpy as np
from models.language_utils import get_word_emb_arr

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        # 用于设置网络中的全连接层
        self.layer_inputx = nn.Linear(dim_in, 512)#(dim_in, 512)
        #激活函数
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0)
        self.layer_hidden1x = nn.Linear(512, 256)#(512, 256)
        self.layer_hidden2x = nn.Linear(256, 64)

        self.layer_inputy = nn.Linear(dim_in, 512)  # (dim_in, 512)
        self.layer_hidden1y = nn.Linear(512, 256)  # (512, 256)
        self.layer_hidden2y = nn.Linear(256, 64)

        self.layer_out = nn.Linear(128, dim_out)#(64, dim_out)
        self.softmax = nn.Softmax(dim=1)
        self.weight_keys = [['layer_inputx.weight', 'layer_inputx.bias'],
                            ['layer_hidden1x.weight', 'layer_hidden1x.bias'],
                            ['layer_hidden2x.weight', 'layer_hidden2x.bias'],
                            ['layer_inputy.weight', 'layer_inputy.bias'],
                            ['layer_hidden1y.weight', 'layer_hidden1y.bias'],
                            ['layer_hidden2y.weight', 'layer_hidden2y.bias'],
                            ['layer_out.weight', 'layer_out.bias']
                            ]

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        y = x.clone()
        x = self.layer_inputx(x)
        x = self.relu(x)
        y = self.layer_inputy(y)
        y = self.relu(y)

        x = self.layer_hidden1x(x)
        x = self.relu(x)
        y = self.layer_hidden1y(y)
        y = self.relu(y)

        x = self.layer_hidden2x(x)
        x = self.relu(x)
        y = self.layer_hidden2y(y)
        y = self.relu(y)

        z = F.cosine_similarity(x,y)
        z = torch.mean(z)

        z3 = torch.sub(x, y)
        z3 = torch.norm(z3)

        x = torch.cat((x,y),1)
        x = self.layer_out(x)
        return self.softmax(x) , z

class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1x = nn.Conv2d(3, 64, 5)
        self.poolx = nn.MaxPool2d(2, 2)
        self.conv2x = nn.Conv2d(64, 64, 5)
        self.fc1x = nn.Linear(64 * 5 * 5, 120)
        self.fc2x = nn.Linear(120, 64)

        self.conv1y = nn.Conv2d(3, 64, 5)
        self.pooly = nn.MaxPool2d(2, 2)
        self.conv2y = nn.Conv2d(64, 64, 5)
        self.fc1y = nn.Linear(64 * 5 * 5, 120)
        self.fc2y = nn.Linear(120, 64)

        self.fc3 = nn.Linear(128, args.num_classes)#(64, args.num_classes)
        self.cls = args.num_classes
        self.drop = nn.Dropout(0.6)

        self.weight_keys = [['fc1x.weight', 'fc1x.bias'],
                            ['fc2x.weight', 'fc2x.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2x.weight', 'conv2x.bias'],
                            ['conv1x.weight', 'conv1x.bias'],
                            ['fc1y.weight', 'fc1y.bias'],
                            ['fc2y.weight', 'fc2y.bias'],
                            ['conv2y.weight', 'conv2y.bias'],
                            ['conv1y.weight', 'conv1y.bias'],
                            ]

    def forward(self, x):
        y = x.clone()
        x = self.poolx(F.relu(self.conv1x(x)))
        x = self.poolx(F.relu(self.conv2x(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1x(x))
        x = F.relu(self.fc2x(x))
        xm = x.norm()
        #x = x/xm

        y = self.pooly(F.relu(self.conv1y(y)))
        y = self.pooly(F.relu(self.conv2y(y)))
        y = y.view(-1, 64 * 5 * 5)
        y = F.relu(self.fc1y(y))
        y = F.relu(self.fc2y(y))
        z = F.cosine_similarity(x, y)
        z = torch.mean(z)
        x = torch.cat((x, y), 1)

        x = self.drop(self.fc3(x))
        return F.log_softmax(x, dim=1) , z

class CNNCifar100(nn.Module):
    def __init__(self, args):
        super(CNNCifar100, self).__init__()
        self.conv1x = nn.Conv2d(3, 64, 5)
        self.poolx = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.6)
        self.conv2x = nn.Conv2d(64, 128, 5)
        self.fc1x = nn.Linear(128 * 5 * 5, 256)
        self.fc2x = nn.Linear(256, 128)
        self.fc3 = nn.Linear(256, args.num_classes)#(128, args.num_classes)

        self.conv1y = nn.Conv2d(3, 64, 5)
        self.pooly = nn.MaxPool2d(2, 2)
        self.conv2y = nn.Conv2d(64, 128, 5)
        self.fc1y = nn.Linear(128 * 5 * 5, 256)
        self.fc2y = nn.Linear(256, 128)

        self.cls = args.num_classes

        self.weight_keys = [['fc1x.weight', 'fc1x.bias'],
                            ['fc2x.weight', 'fc2x.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2x.weight', 'conv2x.bias'],
                            ['conv1x.weight', 'conv1x.bias'],
                            ['fc1y.weight', 'fc1y.bias'],
                            ['fc2y.weight', 'fc2y.bias'],
                            ['conv2y.weight', 'conv2y.bias'],
                            ['conv1y.weight', 'conv1y.bias']
                            ]

    def forward(self, x):
        y = x.clone()
        x = self.poolx(F.relu(self.conv1x(x)))
        x = self.poolx(F.relu(self.conv2x(x)))
        x = x.view(-1, 128 * 5 * 5)
        x = F.relu(self.fc1x(x))
        x = self.drop((F.relu(self.fc2x(x))))

        y = self.pooly(F.relu(self.conv1y(y)))
        y = self.pooly(F.relu(self.conv2y(y)))
        y = y.view(-1, 128 * 5 * 5)
        y = F.relu(self.fc1y(y))
        y = self.drop((F.relu(self.fc2y(y))))

        z = F.cosine_similarity(x, y)
        z = torch.mean(z)

        x = torch.cat((x, y), 1)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), z

class CNN_FEMNIST(nn.Module):
    def __init__(self, args):
        super(CNN_FEMNIST, self).__init__()
        self.conv1x = nn.Conv2d(1, 4, 5)
        self.poolx = nn.MaxPool2d(2, 2)
        self.conv2x = nn.Conv2d(4, 12, 5)
        self.fc1x = nn.Linear(12 * 4 * 4, 120)
        self.fc2x = nn.Linear(120, 100)

        self.conv1y = nn.Conv2d(1, 4, 5)
        self.pooly = nn.MaxPool2d(2, 2)
        self.conv2y = nn.Conv2d(4, 12, 5)
        self.fc1y = nn.Linear(12 * 4 * 4, 120)
        self.fc2y = nn.Linear(120, 100)

        self.fc3 = nn.Linear(100, args.num_classes)

        self.weight_keys = [['fc1x.weight', 'fc1x.bias'],
                            ['fc2x.weight', 'fc2x.bias'],
                            ['fc3.weight', 'fc3.bias'],
                            ['conv2x.weight', 'conv2x.bias'],
                            ['conv1x.weight', 'conv1x.bias'],
                            ['fc1y.weight', 'fc1y.bias'],
                            ['fc2y.weight', 'fc2y.bias'],
                            ['conv2y.weight', 'conv2y.bias'],
                            ['conv1y.weight', 'conv1y.bias']
                            ]

    def forward(self, x):
        y = x.clone()
        x = self.poolx(F.relu(self.conv1x(x)))
        x = self.poolx(F.relu(self.conv2x(x)))
        x = x.view(-1, 12 * 4 * 4)
        x = F.relu(self.fc1x(x))
        x = F.relu(self.fc2x(x))

        y = self.pooly(F.relu(self.conv1y(y)))
        y = self.pooly(F.relu(self.conv2y(y)))
        y = y.view(-1, 12 * 4 * 4)
        y = F.relu(self.fc1y(y))
        y = F.relu(self.fc2y(y))

        z = F.cosine_similarity(x, y)
        z = torch.mean(z)
        x = torch.cat((x, y), 1)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1), z


class RNNSent(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    Modified by: Hongyi Wang from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
    """

    def __init__(self,args, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, emb_arr=None):
        super(RNNSent, self).__init__()
        VOCAB_DIR = 'models/embs.json'
        emb, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
        self.encoder = torch.tensor(emb).to(args.device)
        
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.fc = nn.Linear(nhid, 10)
        self.decoder = nn.Linear(10, ntoken)

        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.device = args.device

    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        input = torch.transpose(input, 0,1)
        emb = torch.zeros((25,4,300)) 
        for i in range(25):
            for j in range(4):
                emb[i,j,:] = self.encoder[input[i,j],:]
        emb = emb.to(self.device)
        emb = emb.view(300,4,25)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(F.relu(self.fc(output)))
        decoded = self.decoder(output[-1,:,:])
        return decoded.t(), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

