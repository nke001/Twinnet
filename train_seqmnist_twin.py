import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets
from layer_pytorch import *
import time
import click
import numpy
import numpy as np
import os
import random
from itertools import chain
import load
import torch.nn.functional as F
from viz import Logger

# set a bunch of seeds
seed = 1234
rng = np.random.RandomState(seed)


def get_epoch_iterator(nbatch, X, Y=None):
    ndata = X.shape[0]
    samples = rng.permutation(np.arange(ndata))
    for b in range(0, ndata, nbatch):
        idx = samples[b:b + nbatch]
        assert len(idx) == nbatch
        x = X[idx]
        if Y is not None:
            y = Y[idx]
        else:
            y = None
        x = x.reshape((-1, 784)).transpose(1, 0)
        yield (x, y)

def binary_crossentropy(x, p):
    return -torch.sum((torch.log(p + 1e-6) * x +
                       torch.log(1 - p + 1e-6) * (1. - x)), 0)

class Model(nn.Module):
    def __init__(self, rnn_dim, nlayers, dropout=0.):
        super(Model, self).__init__()
        self.rnn_dim = rnn_dim
        self.dropout = dropout
        self.nlayers = nlayers
        self.embed = nn.Embedding(2, 300)
        self.fwd_rnn = nn.LSTM(300, rnn_dim, nlayers, dropout=dropout)
        self.bwd_rnn = nn.LSTM(300, rnn_dim, nlayers, dropout=dropout)
        self.fwd_out = nn.Sequential(
            nn.Linear(rnn_dim, 1),
            nn.Sigmoid())
        self.bwd_out = nn.Sequential(
            nn.Linear(rnn_dim, 1),
            nn.Sigmoid())
        self.fwd_aff = nn.Linear(rnn_dim, rnn_dim)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.rnn_dim).zero_()),
                Variable(weight.new(self.nlayers, bsz, self.rnn_dim).zero_()))

    def save(self, filename):
        state = {
            'nlayers': self.nlayers,
            'rnn_dim': self.rnn_dim,
            'dropout': self.dropout,
            'state_dict': self.state_dict()
        }
        torch.save(state, filename)

    @classmethod
    def load(cls, filename):
        state = torch.load(filename)
        model = Model(
            state['rnn_dim'], state['nlayers'], dropout=state['dropout'])
        model.load_state_dict(state['state_dict'])
        return model

    def rnn(self, x, hidden, forward=True):
        rnn_mod = self.fwd_rnn if forward else self.bwd_rnn
        out_mod = self.fwd_out if forward else self.bwd_out
        bsize = x.size(1)
        # run recurrent model
        x = self.embed(x)
        enc_x = x
        out, hidden = rnn_mod(x, hidden)
        out_2d = out.view(out.size(0) * bsize, self.rnn_dim)
        out_2d = out_mod(out_2d)
        out_2d = out_2d.view(out.size(0), bsize)
        # transform forward with affine
        twin_vis = out
        if forward:
            vis_ = out.view(out.size(0) * bsize, self.rnn_dim)
            vis_ = self.fwd_aff(vis_)
            twin_vis = vis_.view(out.size(0), bsize, self.rnn_dim)
        return out_2d, twin_vis, hidden, enc_x

    def forward(self, fwd_x, bwd_x, hidden):
        fwd_out, fwd_vis, _, _ = self.rnn(fwd_x, hidden)
        bwd_out, bwd_vis, _, _ = self.rnn(bwd_x, hidden, forward=False)
        return fwd_out, bwd_out, fwd_vis, bwd_vis


def evaluate(model, bsz, data_x, data_y):
    model.eval()
    hidden = model.init_hidden(bsz)
    valid_loss = []
    for x, _ in get_epoch_iterator(bsz, data_x, data_y):
        x = np.concatenate([np.zeros((1, bsz)).astype('int32'), x], axis=0)
        x = torch.from_numpy(x)
        inp = Variable(x[:-1], volatile=True).long().cuda()
        trg = Variable(x[1:], volatile=True).float().cuda()
        ret = model.rnn(inp, hidden)
        loss = binary_crossentropy(trg, ret[0]).mean()
        valid_loss.append(loss.data[0])
    return np.asarray(valid_loss).mean()


@click.command()
@click.option('--expname', default='mnist_logs')
@click.option('--logdir', default=None)
@click.option('--modeldir', default=None)
@click.option('--datadir', default='./mnist/data')
@click.option('--nlayers', default=2)
@click.option('--dropout', default=0.0)
@click.option('--num_epochs', default=50)
@click.option('--rnn_dim', default=512)
@click.option('--bsz', default=20)
@click.option('--lr', default=0.001)
@click.option('--twin', default=0.)
@click.option('--dont_disconnect', is_flag=True)
def train(expname, logdir, modeldir, datadir, nlayers, dropout, num_epochs, rnn_dim, bsz, lr, twin, dont_disconnect):
    # use hugo's binarized MNIST
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    log_interval = 100
    model_id = 'mnist_twin{}_nl{}_dim{}_drop{}_ddis{}_seed{}'.format(
            twin, nlayers, rnn_dim, dropout, dont_disconnect, seed)
    if logdir is None or modeldir is None:
        logdir = expname
        modeldir = expname
        if not os.path.exists(expname):
            os.makedirs(expname)
    log_file_name = os.path.join(logdir, model_id + '.txt')
    model_file_name = os.path.join(modeldir, model_id + '.pt')
    pkl_file_name = os.path.join(logdir, model_id + '.pkl') 
    logger = Logger(pkl_file_name)
    log_file = open(log_file_name, 'w')

    # Hugo's version, for compatibility with SOTA.
    train_x, valid_x, test_x = load.load_binarized_mnist(datadir)
    train_y = None
    valid_y = None
    test_y = None

    # First example looks like...
    print(train_x[0])

    model = Model(rnn_dim, nlayers, dropout)
    model.cuda()
    hidden = model.init_hidden(bsz)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    idx = np.arange(784)[::-1].tolist()
    idx = torch.LongTensor(idx)
    idx = Variable(idx).cuda()

    nbatches = train_x.shape[0] // bsz
    t = time.time()
    for epoch in range(num_epochs):
        step = 0
        old_valid_loss = np.inf
        b_fwd_loss, b_bwd_loss, b_twin_loss, b_all_loss = 0., 0., 0., 0.
        model.train()
        print('Epoch {}: ({})'.format(epoch, model_id.upper()))
        for x, _ in get_epoch_iterator(bsz, train_x, train_y):
            opt.zero_grad()
            x = Variable(torch.from_numpy(x)).long().cuda()
            x_ = torch.cat((x[:1] * 0, x), 0)
            assert x_.size(0) == 785
            fwd_inp = x_[:-1]
            fwd_trg = x_[1:].float()

            bx_ = x.index_select(0, idx)
            bx_ = torch.cat((x[:1] * 0, bx_), 0) 
            assert bx_.size(0) == 785
            bwd_inp = bx_[:-1]
            bwd_trg = bx_[1:].float()

            # compute all the states for forward and backward
            fwd_out, bwd_out, fwd_vis, bwd_vis = \
                    model(fwd_inp, bwd_inp, hidden)
            assert fwd_out.size(0) == 784
            assert bwd_out.size(0) == 784
            assert fwd_vis.size(0) == 784
            assert bwd_vis.size(0) == 784
            fwd_loss = binary_crossentropy(fwd_trg, fwd_out).mean()
            bwd_loss = binary_crossentropy(bwd_trg, bwd_out).mean()
            bwd_loss = bwd_loss * (twin > 0.)

            bwd_vis_inv = bwd_vis.index_select(0, idx)
            # interrupt gradient here
            if dont_disconnect is False:
                bwd_vis_inv = bwd_vis_inv.detach()

            # mean over batch, over dimensions
            twin_loss = ((fwd_vis - bwd_vis_inv) ** 2).sum(0).mean()
            twin_loss = twin_loss * twin
            all_loss = fwd_loss + bwd_loss + twin_loss
            all_loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), 5.)
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
                info = {
                    'fwd_loss/train': b_fwd_loss / log_interval,
                    'bwd_loss/train': b_bwd_loss / log_interval,
                    'twin_loss/train': b_twin_loss / log_interval
                }
                for tag, value in info.items():
                    logger.scalar_summary(tag, value)
                logger.flush()

                b_all_loss = 0.
                b_fwd_loss = 0.
                b_bwd_loss = 0.
                b_twin_loss = 0.
                t = time.time()
                print(log_line)
                log_file.write(log_line + '\n')
                log_file.flush()

            step += 1

        # evaluate per epoch
        print('--- Epoch finished ----')
        val_loss = evaluate(model, bsz, valid_x, valid_y)
        log_line = 'valid -- nll: %f' % (val_loss)
        print(log_line)
        log_file.write(log_line + '\n')
        test_loss = evaluate(model, bsz, test_x, test_y)
        log_line = 'test -- nll: %f' % (test_loss)
        print(log_line)
        log_file.write(log_line + '\n')
        log_file.flush()

        info = {
                'fwd_loss/valid': val_loss,
                'fwd_loss/test': test_loss
                }
        for tag, value in info.items():
            logger.scalar_summary(tag, value)
        logger.flush()

        if old_valid_loss > val_loss:
            old_valid_loss = val_loss
            model.save(model_file_name)

        if epoch in [5, 10, 15]:
            for param_group in opt.param_groups:
                lr = param_group['lr']
                lr *= 0.5
                param_group['lr'] = lr


if __name__ == '__main__':
    train()
