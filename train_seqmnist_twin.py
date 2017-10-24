import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets
from layer_pytorch import *
import time
from char_data_iterator import TextIterator
import numpy
import numpy as np
import os
import random
from itertools import chain
import load


length = 784
rnn_dim = 512
bsz = 50
num_epochs = 40
lr = 0.001
n_words = 2
log_interval = 100
twin = 0.5
seed = 1234

folder_id = 'mnist_twin_logs'
model_id = 'mnist_twin{}'.format(twin)
file_name = os.path.join(folder_id, model_id + '.txt')
model_file_name = os.path.join(folder_id, model_id + '.pt')

log_file = open(file_name, 'w')
rng = np.random.RandomState(seed)


def binarize(rng, x):
    return (x > rng.rand(x.shape[0], x.shape[1])).astype('int32')


train_x, valid_x, test_x, train_y, valid_y, test_y = \
    load.load_mnist('./mnist/data')
train_x = binarize(rng, train_x)
valid_x = binarize(rng, valid_x)
test_x = binarize(rng, test_x)
print(train_x[0])


def get_epoch_iterator(nbatch, X, Y):
    ndata = X.shape[0]
    samples = rng.permutation(np.arange(ndata))
    for b in range(0, ndata, nbatch):
        idx = samples[b:b + nbatch]
        assert len(idx) == nbatch
        x = X[idx]
        y = Y[idx]
        x = x.reshape((-1, 784)).transpose(1, 0)
        yield (x, y)


def binary_crossentropy(x, p):
    return -torch.sum((torch.log(p + 1e-6) * x + \
                       torch.log(1 - p + 1e-6) * (1. - x)), 0)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embed = nn.Embedding(2, 200)
        self.fwd_rnn = nn.LSTM(200, rnn_dim, 1, batch_first=False, dropout=0)
        self.bwd_rnn = nn.LSTM(200, rnn_dim, 1, batch_first=False, dropout=0)
        self.fwd_out = nn.Sequential(nn.Linear(rnn_dim, 1), nn.Sigmoid())
        self.bwd_out = nn.Sequential(nn.Linear(rnn_dim, 1), nn.Sigmoid())

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(1, bsz, rnn_dim).zero_()),
                Variable(weight.new(1, bsz, rnn_dim).zero_()))
    
    def rnn(self, x, hidden, forward=True):
        rnn_mod = self.fwd_rnn if forward else self.bwd_rnn
        out_mod = self.fwd_out if forward else self.bwd_out
        bsize = x.size(1)
        x = self.embed(x)
        vis, states = rnn_mod(x, hidden)
        vis_ = vis.view(vis.size(0) * bsize, rnn_dim)
        out = out_mod(vis_)
        out = out.view(vis.size(0), bsize)
        return out, vis

    def forward(self, fwd_x, bwd_x, hidden):
        fwd_out, fwd_vis = self.rnn(fwd_x, hidden)
        bwd_out, bwd_vis = self.rnn(bwd_x, hidden, forward=False)
        return fwd_out, bwd_out, fwd_vis, bwd_vis


model = Model()
model.cuda()
hidden = model.init_hidden(bsz)
opt = torch.optim.Adam(model.parameters(), lr=lr)


def evaluate(data_x, data_y):
    valid_loss = []
    for x, _ in get_epoch_iterator(bsz, data_x, data_y):
        x = torch.from_numpy(x)
        inp = Variable(x[:-1], volatile=True).long().cuda()
        trg = Variable(x[1:], volatile=True).float().cuda()
        opt.zero_grad()
        out, sta = model.rnn(inp, hidden)
        loss = binary_crossentropy(trg, out).mean()
        valid_loss.append(loss.data[0])
    return np.asarray(valid_loss).mean()


nbatches = train_x.shape[0] // bsz
t = time.time()
for epoch in range(num_epochs):
    step = 0
    train_len = train_x.shape[0]

    b_fwd_loss, b_bwd_loss, b_twin_loss, b_all_loss = 0., 0., 0., 0. 
    print('Epoch {}: ({})'.format(epoch, model_id.upper()))
    for x, _ in get_epoch_iterator(50, train_x, train_y):
        opt.zero_grad()
        # x = (x1, x2, x3, x4)
        # fwd_inp = (x1, x2, x3)
        # fwd_trg = (x2, x3, x4)
        fwd_x = torch.from_numpy(x)
        fwd_inp = Variable(fwd_x[:-1]).long().cuda()
        fwd_trg = Variable(fwd_x[1:]).float().cuda()
        # bwd_x = (x4, x3, x2, x1)
        # bwd_inp = (x4, x3, x2)
        # bwd_trg = (x3, x2, x1)
        bwd_x = numpy.flip(x, 0).copy()
        bwd_x = torch.from_numpy(bwd_x)
        bwd_inp = Variable(bwd_x[:-1]).long().cuda()
        bwd_trg = Variable(bwd_x[1:]).float().cuda()
        
        # compute all the states for forward and backward
        fwd_out, bwd_out, fwd_vis, bwd_vis = model(fwd_inp, bwd_inp, hidden)
        fwd_loss = binary_crossentropy(fwd_trg, fwd_out).mean()
        bwd_loss = binary_crossentropy(bwd_trg, bwd_out).mean()
        bwd_loss = bwd_loss * (twin > 0.)

        # reversing backstates
        # fwd_vis = (out_x2, out_x3, out_x4)
        # bwd_vis_inv = (out_x1, out_x2, out_x3)
        # therefore match: fwd_vis[:-1] and bwd_vis_inv[1:]
        idx = np.arange(bwd_vis.size(0))[::-1].tolist()
        idx = torch.LongTensor(idx)
        idx = Variable(idx).cuda()
        bwd_vis_inv = bwd_vis.index_select(0, idx)
        bwd_vis_inv = bwd_vis_inv.detach()
        
        twin_loss = ((fwd_vis[:-1] - bwd_vis_inv[1:]) ** 2).mean()
        twin_loss = twin_loss * twin
        all_loss = fwd_loss + bwd_loss + twin_loss
        all_loss.backward()
        opt.step()

        b_all_loss += all_loss.data[0]
        b_fwd_loss += fwd_loss.data[0]
        b_bwd_loss += bwd_loss.data[0]
        b_twin_loss += twin_loss.data[0]

        if (step + 1) % log_interval == 0:
            s = time.time()
            log_line = 'Epoch [%d/%d], Step [%d/%d], loss: %f, fwd loss: %f, twin loss: %f, bwd loss: %f, %.2fit/s' % (
                epoch, num_epochs, step + 1, nbatches,
                b_all_loss / log_interval,
                b_fwd_loss / log_interval,
                b_twin_loss / log_interval,
                b_bwd_loss / log_interval,
                log_interval / (s - t))
            b_all_loss = 0.
            b_fwd_loss = 0.
            b_bwd_loss = 0.
            b_twin_loss = 0.
            t = time.time()
            print(log_line)
            log_file.write(log_line + '\n')

        step += 1

    # evaluate per epoch
    print('--- Epoch finished ----')
    val_loss = evaluate(valid_x, valid_y)
    log_line = 'valid -- nll: %f' % (val_loss)
    print(log_line)
    log_file.write(log_line + '\n')
    test_loss = evaluate(test_x, test_y)
    log_line = 'test -- nll: %f' % (test_loss)
    print(log_line)
    log_file.write(log_line + '\n')

