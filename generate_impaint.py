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
import scipy
from train_impaint_twin import Model


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
@click.option('--visibility', type=float)
def generate(filename, visibility):
    seed = 1234
    print('Loading model from {}'.format(filename))
    model = Model.load(filename)
    print('DONE.')
    train_x, valid_x, test_x = \
            load.load_binarized_mnist('./mnist/data')
    images = train_x[:8].T
    hidden = model.init_hidden(8)

    npixels_visible = int(visibility * 784)
    npixels_hidden = int((1. - visibility) * 784) 
    
    idx = np.arange(npixels_hidden)[::-1].tolist()
    idx = torch.LongTensor(idx)
    idx = Variable(idx)

    x = Variable(torch.from_numpy(images)).long()
    vis_x = x[:npixels_visible]
    hid_x = x[npixels_visible:]

    x_ = torch.cat((hid_x[:1] * 0, hid_x), 0)
    fwd_inp = x_[:-1]
    fwd_trg = x_[1:].float()
            
    # invert pixels for backward pass
    bx_ = hid_x.index_select(0, idx)
    bx_ = torch.cat((hid_x[:1] * 0, bx_), 0)
    bwd_inp = bx_[:-1]
    bwd_trg = bx_[1:].float()

    # compute all the states for forward and backward
    fwd_out, bwd_out, fwd_vis, bwd_vis = \
            model(vis_x, fwd_inp, bwd_inp, hidden)
    bwd_out_inv = bwd_out.index_select(0, idx)
    bwd_vis_inv = bwd_vis.index_select(0, idx)
    # mean over batch, over dimensions
    twin_loss = ((fwd_vis - bwd_vis_inv) ** 2).mean(2)
    twin_loss = twin_loss.data.cpu().numpy()
    twin_loss = normalize(twin_loss, axis=1)

    # original images
    imgs_out = images.T.reshape((8, 28, 28))
    # twin loss
    loss_out = np.concatenate([images[:npixels_visible], twin_loss], axis=0) 
    loss_out = loss_out.T.reshape((8, 28, 28))
    # forward rec 
    fwd_out = np.concatenate([images[:npixels_visible], fwd_out.data.cpu().numpy()], axis=0) 
    fwd_out = fwd_out.T.reshape((8, 28, 28))
    # backward rec
    bwd_out = np.concatenate([images[:npixels_visible], bwd_out_inv.data.cpu().numpy()], axis=0) 
    bwd_out = bwd_out.T.reshape((8, 28, 28))
    # concat everything
    all_out = numpy.concatenate([imgs_out, loss_out, fwd_out, bwd_out], axis=0)
    grayscale_grid_vis(all_out, 4, 8, '{}_gen.png'.format(filename))

generate()
