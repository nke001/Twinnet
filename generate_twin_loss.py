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


def normalize(x, axis=0):
    x = x - np.min(x, axis=axis, keepdims=True)
    x = x / (np.max(x, axis=axis, keepdims=True) + 1e-5)
    return x


@click.command()
@click.option('--filename')
def generate(filename):
    seed = 1234
    print('Loading model from {}'.format(filename))
    model = Model.load(filename)
    print('DONE.')
    train_x, valid_x, test_x = \
        load.load_binarized_mnist('./data/mnist')
    train_x = train_x[:4].T
    hidden = model.init_hidden(4)

    x = train_x
    x = Variable(torch.from_numpy(x)).long()
    x_ = torch.cat((x[:1] * 0, x), 0)

    idx = np.arange(784)[::-1].tolist()
    idx = torch.LongTensor(idx)
    idx = Variable(idx)
    fwd_inp = x_[:-1]
    bx_ = x_.index_select(0, idx)
    bx_ = torch.cat((x[:1] * 0, x), 0)
    bwd_inp = bx_[:-1]

    fwd_out, bwd_out, fwd_vis, bwd_vis = \
        model(fwd_inp, bwd_inp, hidden)
    assert fwd_out.size(0) == 784
    assert bwd_out.size(0) == 784
    assert fwd_vis.size(0) == 784
    assert bwd_vis.size(0) == 784

    bwd_vis_inv = bwd_vis.index_select(0, idx)
    # interrupt gradient here
    bwd_vis_inv = bwd_vis_inv.detach()
    # mean over batch, over dimensions
    twin_loss = ((fwd_vis - bwd_vis_inv) ** 2).mean(2)
    twin_loss = twin_loss.data.cpu().numpy()

    twin_loss = twin_loss.T.reshape((4, 28, 28))
    examples = train_x.reshape((4, 28, 28))
    images = numpy.concatenate([examples, twin_loss], axis=0)
    grayscale_grid_vis(images, 4, 2, '{}_gen.png'.format(filename))

generate()