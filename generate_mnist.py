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

    print(np.concatenate(outs, 1))

generate()