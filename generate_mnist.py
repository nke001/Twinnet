import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import click
import numpy
import numpy as np
import os
import random
import load
from train_seqmnist_twin import Model


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def grayscale_grid_vis(X, nh, nw, save_path=None):
    h, w = X[0].shape[:2]
    h = h + 2  # make room for  a little border
    w = w + 2
    x_shell = np.zeros((h, w)) + ((np.max(X) - np.min(X)) / 2.)
    img = np.zeros((h * nh, w * nw))
    for n, x in enumerate(X):
        j = n // nw
        i = n % nw
        x_shell[1:-1, 1:-1] = x[:, :]
        img[(j * h):(j * h + h), (i * w):(i * w + w)] = x_shell[:, :]
    if save_path is not None:
        scipy.misc.imsave(save_path, img)
    return img

@click.command()
@click.option('--filename')
def generate(filename):
    seed = 1234
    rng = np.random.RandomState(seed)
    print('Loading model from {}'.format(filename))
    model = Model.load(filename)
    print('DONE.')
    hidden = model.init_hidden(16)
    x = np.zeros((1, 16)).astype('int32')
    outs = [x]
    for i in range(784):
        print('Generating pixel... {}'.format(i))
        last_x = Variable(torch.from_numpy(outs[-1]))
        out, vis, sta, _ = model.rnn(last_x, hidden)
        out = (out.cpu()).data
        smp = (out > rng.rand(out.shape)).astype('int32')
        outs.append(smp)
        hidden = repackage_hidden(sta)
    outs = outs[1:]
    outs = np.concatenate(outs, 0).T
    outs = outs.reshape((16, 28, 28))
    grayscale_grid_vis(outs, 4, 4, '{}_gen.png'.format(filename))
generate()
