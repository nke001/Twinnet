import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets
from layer_pytorch import *
import time
from char_data_iterator import TextIterator
import numpy
import os
import random


length = 784
input_size = 1
rnn_dim = 512
num_layers = 2
num_classes = 2
batch_size = 32
valid_batch_size = 32
num_epochs = 20
lr = 0.0001
n_words=2
maxlen=785
dataset='/data/lisatmp4/kenan/dataset/binarized_mnist/structured/train.txt'
valid_dataset='/data/lisatmp4/kenan/dataset/binarized_mnist/structured/test.txt'
dictionary='dict_bin_mnist.npz'
sequence_length = 28
truncate_length = 10
attn_every_k = 10



file_name = 'mnist_logs/mnist_lstm_' + str(random.randint(1000,9999)) + '.txt'


train = TextIterator(dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen,
                         minlen=length)
valid = TextIterator(valid_dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen,
                         minlen=length)


rnn = RNN_LSTM(input_size, rnn_dim, num_layers, num_classes)

rnn.cuda()

criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(rnn.parameters(), lr=lr)

def evaluate_valid(valid):
    valid_loss = []
    valid_acc = []
    i = 0
    for x in valid:
        x = numpy.asarray(x, dtype=numpy.float32)
        x = torch.from_numpy(x)
        x = x.view(x.size()[0], x.size()[1], input_size)
        y = torch.cat(( x[:, 1:, :], torch.zeros([x.size()[0], 1, input_size])), 1)
        images = Variable(x).cuda()
        labels = Variable(y).long().cuda()
        opt.zero_grad()
        outputs= rnn(images)
        shp = outputs.size()
        outputs_reshp = outputs.view([shp[0] * shp[1], num_classes])
        labels_reshp = labels.view(shp[0] * shp[1])
        loss = criterion(outputs_reshp, labels_reshp)
        acc =  (outputs.max(dim=2)[1] - labels).abs().sum()
        
        acc = float(acc.data[0]) / (batch_size * 784 )
        valid_acc.append(acc)
        valid_loss.append(784 * float(loss.data[0]))
        i += 1
    log_line = 'Epoch [%d/%d],  average Loss: %f, average accuracy %f, validation ' %(epoch, num_epochs,  numpy.asarray(valid_loss).mean(), 1.0 - numpy.asarray(valid_acc).mean())
    print  (log_line)
    with open(file_name, 'a') as f:
        f.write(log_line)


for epoch in range(num_epochs):
    i = 0
    for x in train:
        t = -time.time()

        x = numpy.asarray(x, dtype=numpy.float32)
        x = torch.from_numpy(x)
        x = x.view(x.size()[0], x.size()[1], input_size)
        y = torch.cat(( x[:, 1:, :], torch.zeros([x.size()[0], 1, input_size])), 1)
        images = Variable(x).cuda()
        labels = Variable(y).long().cuda()
        opt.zero_grad()
        outputs = rnn(images)
        shp = outputs.size()
        outputs_reshp = outputs.view([shp[0] * shp[1], num_classes])
        labels_reshp = labels.view(shp[0] * shp[1])
        loss = criterion(outputs_reshp, labels_reshp)
        loss.backward()
        opt.step()

        t += time.time()

        if (i+1) % 10 == 0:
            log_line = 'Epoch [%d/%d], Step %d, Loss: %f, batch_time: %f \n' %(epoch, num_epochs, i+1, 784 * loss.data[0], t)
            print (log_line)
            with open(file_name, 'a') as f:
                f.write(log_line)


        if (i + 1) % 100 == 0:
            evaluate_valid(valid)

        i += 1

    # evaluate per epoch
    print '--- Epoch finished ----'
    evaluate_valid(valid)






























