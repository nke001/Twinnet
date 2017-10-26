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

# set a bunch of seeds
seed = 1234
rng = np.random.RandomState(seed)


def get_epoch_iterator(nbatch, X, Y=None, show=0.5):
    ndata = X.shape[0]
    samples = rng.permutation(np.arange(ndata))

    for b in range(0, ndata, nbatch):
        idx = samples[b:b + nbatch]
        assert len(idx) == nbatch
        x = X[idx]
        if Y is not None:
            y = Y[idx]
            y = np.eye(10)[y]
        else:
            y = None
        x = x.reshape((-1, 784)).transpose(1, 0)
        # occlude x
        assert show in [0.25, 0.5, 0.75]
        pix_vis = int(show * 784)
        x_vis = x[:pix_vis]
        x_hid = x[pix_vis:]
        yield (x_vis, x_hid)


def binary_crossentropy(x, p):
    return -torch.sum((torch.log(p + 1e-6) * x +
                       torch.log(1 - p + 1e-6) * (1. - x)), 0)


class MyLSTM(nn.Module):
    def __init__(self, ninp, rnn_dim, nlayers):
        super(MyLSTM, self).__init__()
        self.rnns = []
        for i in range(nlayers):
            self.rnns.append(nn.LSTM(ninp if i == 0 else rnn_dim, rnn_dim, 1))
        self.rnns = nn.ModuleList(self.rnns)

    def forward(self, x, hidden):
        length = x.size(0)
        nlayers = len(self.rnns)
        inputs = [x]
        states = []
        output = []
        for i in range(nlayers):
            vis, hid = self.rnns[i](inputs[i], (
                hidden[0][i].unsqueeze(0),
                hidden[1][i].unsqueeze(0)))
            states.append(hid)
            output.append(vis)
            inputs.append(vis)
        hidden = (torch.cat([s[0] for s in states], 0),
                  torch.cat([s[1] for s in states], 0))
        return output[0], torch.cat(output, 0), hidden


class Model(nn.Module):
    def __init__(self, rnn_dim, nlayers, deep_out=True):
        super(Model, self).__init__()
        self.rnn_dim = rnn_dim
        self.nlayers = nlayers
        self.deep_out = deep_out
        self.embed = nn.Embedding(2, 300)
        self.fwd_rnn = MyLSTM(300, rnn_dim, nlayers)
        self.bwd_rnn = MyLSTM(300, rnn_dim, nlayers)
        if self.deep_out:
            # additional layer before the output
            self.fwd_prj_prev = nn.Linear(300, 512)
            self.bwd_prj_prev = nn.Linear(300, 512)
            self.fwd_prj_out = nn.Linear(rnn_dim, 512)
            self.bwd_prj_out = nn.Linear(rnn_dim, 512)
        self.fwd_out = nn.Sequential(
            nn.Linear(512 if deep_out else rnn_dim, 1),
            nn.Sigmoid())
        self.bwd_out = nn.Sequential(
            nn.Linear(512 if deep_out else rnn_dim, 1),
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
            'deep_out': self.deep_out,
            'state_dict': self.state_dict()
        }
        torch.save(state, filename)

    @classmethod
    def load(cls, filename):
        state = torch.load(filename)
        model = Model(
            state['rnn_dim'], state['nlayers'],
            deep_out=state['deep_out'])
        model.load_state_dict(state['state_dict'])
        return model

    def rnn(self, x, hidden, forward=True):
        rnn_mod = self.fwd_rnn if forward else self.bwd_rnn
        out_mod = self.fwd_out if forward else self.bwd_out
        bsize = x.size(1)
        # run recurrent model
        x = self.embed(x)
        enc_x = x
        out, vis, hidden = rnn_mod(x, hidden)
        out_2d = out.view(out.size(0) * bsize, self.rnn_dim)
        # compute deep output layer or simple output
        if self.deep_out:
            x_ = x.view(out.size(0) * bsize, x.size(2))
            prv_mod = self.fwd_prj_prev if forward else self.bwd_prj_prev
            prj_mod = self.fwd_prj_out if forward else self.bwd_prj_out
            out_2d = F.leaky_relu(prv_mod(x_) + prj_mod(out_2d), 0.3, False)
        out_2d = out_mod(out_2d)
        out_2d = out_2d.view(out.size(0), bsize)
        # transform forward with affine
        twin_vis = vis
        if forward:
            vis_ = vis.view(vis.size(0) * bsize, self.rnn_dim)
            vis_ = self.fwd_aff(vis_)
            twin_vis = vis_.view(vis.size(0), bsize, self.rnn_dim)
        return out_2d, twin_vis, hidden, enc_x

    def forward(self, vis_x, hid_x_fwd, hid_x_bwd, hidden):
        #
        fwd_out, fwd_vis, fwd_hid, _ = self.rnn(vis_x, hidden)
        bwd_out, bwd_vis, bwd_hid, _ = self.rnn(vis_x, hidden, forward=False)
        # decode
        fwd_out, fwd_vis, _, _ = self.rnn(hid_x_fwd, fwd_hid)
        bwd_out, bwd_vis, _, _ = self.rnn(hid_x_bwd, bwd_hid, forward=False)
        return fwd_out, bwd_out, fwd_vis, bwd_vis


def evaluate(model, bsz, data, visibility=0.5):
    model.eval()
    hidden = model.init_hidden(bsz)
    valid_loss = []
    for vis_x, hid_x in get_epoch_iterator(bsz, data, show=visibility):
        vis_x = Variable(torch.from_numpy(vis_x)).long().cuda()
        x_ = np.concatenate([
            np.zeros((1, bsz)).astype('int32'), hid_x], axis=0)
        x = torch.from_numpy(x_)
        inp = Variable(x[:-1]).long().cuda()
        trg = Variable(x[1:]).float().cuda()
        ret = model.rnn(vis_x, hidden)
        ret = model.rnn(fwd_inp, ret[-1])
        loss = binary_crossentropy(fwd_trg, ret[0]).mean()
        valid_loss.append(loss.data[0])
    return np.asarray(valid_loss).mean()


@click.command()
@click.option('--expname', default='inpaint_logs')
@click.option('--nlayers', default=3)
@click.option('--visibility', default=0.5)
@click.option('--num_epochs', default=50)
@click.option('--rnn_dim', default=512)
@click.option('--deep_out', is_flag=True)
@click.option('--bsz', default=20)
@click.option('--lr', default=0.0002)
@click.option('--twin', default=0.)
def train(expname, nlayers, visibility, num_epochs,
          rnn_dim, deep_out, bsz, lr, twin):
    assert not deep_out
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    npixels_visible = int(visibility * 784)
    print('Pixels visible: {}'.format(npixels_visible))
    log_interval = 100
    model_id = 'inpaint_twin{}_do{}_nl{}_dim{}_vis{}'.format(
        twin, deep_out, nlayers, rnn_dim, visibility)
    if not os.path.exists(expname):
        os.makedirs(expname)
    log_file_name = os.path.join(expname, model_id + '.txt')
    model_file_name = os.path.join(expname, model_id + '.pt')
    log_file = open(log_file_name, 'w')

    # "home-made" binarized MNIST version. Use with fixed binarization
    # during training.
    def binarize(rng, x):
        return (x > rng.rand(x.shape[0], x.shape[1])).astype('int32')

    train_x, valid_x, test_x, train_y, valid_y, test_y = \
        load.load_mnist('./mnist/data')
    train_x = binarize(rng, train_x)
    valid_x = binarize(rng, valid_x)
    test_x = binarize(rng, test_x)
    # First example looks like...
    print(train_x[0])

    model = Model(rnn_dim, nlayers, deep_out=deep_out)
    model.cuda()
    hidden = model.init_hidden(bsz)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    nbatches = 500
    t = time.time()
    for epoch in range(num_epochs):
        step = 0
        old_valid_loss = np.inf
        b_fwd_loss, b_bwd_loss, b_twin_loss, b_all_loss = 0., 0., 0., 0.
        model.train()

        print('Epoch {}: ({})'.format(epoch, model_id.upper()))
        for vis_x, hid_x in get_epoch_iterator(bsz, train_x, show=visibility):
            vis_inp = Variable(torch.from_numpy(vis_x)).long().cuda()
            x_ = np.concatenate([
                np.zeros((1, bsz)).astype('int32'), hid_x], axis=0)
            x = torch.from_numpy(x_)
            fwd_inp = Variable(x[:-1]).long().cuda()
            fwd_trg = Variable(x[1:]).float().cuda()

            bwd_x = numpy.flip(hid_x, 0).copy()
            bwd_x = np.concatenate([
                np.zeros((1, bsz)).astype('int32'), bwd_x], axis=0)
            bwd_x = torch.from_numpy(bwd_x)
            bwd_inp = Variable(bwd_x[:-1]).long().cuda()
            bwd_trg = Variable(bwd_x[1:]).float().cuda()

            # compute all the states for forward and backward
            fwd_out, bwd_out, fwd_vis, bwd_vis = \
                model(vis_inp, fwd_inp, bwd_inp, hidden)
            assert fwd_out.size(0) == npixels_visible
            assert bwd_out.size(0) == npixels_visible
            fwd_loss = binary_crossentropy(fwd_trg, fwd_out).mean()
            bwd_loss = binary_crossentropy(bwd_trg, bwd_out).mean()
            bwd_loss = bwd_loss * (twin > 0.)

            idx = np.arange(bwd_vis.size(0))[::-1].tolist()
            idx = torch.LongTensor(idx)
            idx = Variable(idx).cuda()
            bwd_vis_inv = bwd_vis.index_select(0, idx)
            # interrupt gradient here
            bwd_vis_inv = bwd_vis_inv.detach()

            # mean over batch, over dimensions
            twin_loss = ((fwd_vis - bwd_vis_inv) ** 2).mean(2)
            twin_loss = twin_loss.mean(1)
            # sum over timesteps (ratio number of layers)
            twin_loss = twin_loss.sum(0) / nlayers
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
        val_loss = evaluate(model, bsz, seq_width)
        log_line = 'valid -- epoch %s, cost %f' % (epoch, val_loss)
        print(log_line)
        log_file.write(log_line + '\n')
        test_loss = evaluate(model, bsz, seq_width)
        log_line = 'test -- epoch %s, cost %f' % (epoch, test_loss)
        print(log_line)
        log_file.write(log_line + '\n')
        log_file.flush()

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
