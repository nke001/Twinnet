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
from itertools import chain


length = 784
input_size = 1
rnn_dim = 512
num_layers = 2
num_classes = 2
batch_size = 50
valid_batch_size = 32
num_epochs = 40
lr = 0.0005
n_words=2
maxlen=785
dataset = 'bin_mnist.npy'
truncate_length = 10
attn_every_k = 10
embed_size = 256

folder_id = 'mnist_twin_logs'
model_id = 'mnist_twin' + str(random.randint(1000,9999))
#os.mkdir(folder_id)
file_name = os.path.join(folder_id, model_id + '.txt')
model_file_name = os.path.join(folder_id, model_id + '.pkl')
hist_valid_loss = 1.0

#file_name = 'mnist_logs/mnist_lstm_lr_' +  str(lr) + str(random.randint(1000,9999)) + '.txt'


'''train = TextIterator(dataset,
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
'''
data = numpy.load('bin_mnist.npy')

def prepare_data (data, batch_size):
    train_x = data.item().get('train_set')
    train_y = data.item().get('train_labels')
    valid_x = data.item().get('valid_set')
    valid_y = data.item().get('valid_labels')
    
    shp = train_x.shape
    train_x = train_x.reshape(shp[0]/ batch_size, batch_size, shp[1])
    shp = train_y.shape
    train_y = train_y.reshape(shp[0]/ batch_size, batch_size)
    shp = valid_x.shape
    valid_x = valid_x.reshape(shp[0]/ batch_size, batch_size, shp[1])
    shp = valid_y.shape
    valid_y = valid_y.reshape(shp[0]/ batch_size, batch_size)

    return (train_x, train_y, valid_x, valid_y)

train_x, train_y, valid_x, valid_y = prepare_data(data, batch_size)


rnn = RNN_LSTM_embed_twin(input_size, embed_size, rnn_dim, num_layers, num_classes)

back_rnn = RNN_LSTM_embed_twin(input_size, embed_size, rnn_dim, num_layers, num_classes, reverse=True)

rnn.cuda()
back_rnn.cuda()

criterion = nn.CrossEntropyLoss()
l2_criterion = nn.MSELoss()

all_param = chain(rnn.parameters(), back_rnn.parameters())
opt = torch.optim.Adam(all_param, lr=lr)

def evaluate_valid(valid_x):
    valid_loss = []
    valid_acc = []
    i = 0
    valid_len = valid_x.shape[0]
    for i in range(valid_len):
        x = valid_x[i]
        x = numpy.asarray(x, dtype=numpy.float32)
        x = torch.from_numpy(x)
        x = x.view(x.size()[0], x.size()[1], input_size)
        y = torch.cat(( x[:, 1:, :], torch.zeros([x.size()[0], 1, input_size])), 1)
        images = Variable(x).cuda()
        labels = Variable(y).long().cuda()
        opt.zero_grad()
        outputs, states= rnn(images)
        shp = outputs.size()
        outputs_reshp = outputs.view([shp[0] * shp[1], num_classes])
        labels_reshp = labels.view(shp[0] * shp[1])
        loss = criterion(outputs_reshp, labels_reshp)
        acc =  (outputs.max(dim=2)[1] - labels).abs().sum()
        
        acc = float(acc.data[0]) / (batch_size * 784 )
        valid_acc.append(acc)
        valid_loss.append(784 * float(loss.data[0]))
        i += 1
    log_line = 'MNIST generation Epoch [%d/%d],  average Loss: %f, average accuracy %f, validation ' %(epoch, num_epochs,  numpy.asarray(valid_loss).mean(), 1.0 - numpy.asarray(valid_acc).mean())
    print  (log_line)
    with open(file_name, 'a') as f:
        f.write(log_line)


for epoch in range(num_epochs):
    step = 0
    train_len = train_x.shape[0]
    for step in range(train_len):
        t = -time.time()
        x = train_x[step]
        x = numpy.asarray(x, dtype=numpy.float32)
        
        
        back_x = numpy.flip(x,1).copy()
        back_x = torch.from_numpy(back_x)
        back_x = back_x.view(back_x.size()[0], back_x.size()[1], input_size)
        #back_y = torch.cat(( back_x[:, 1:, :], torch.zeros([back_x.size()[0], 1, input_size])), 1)
        back_y = back_x[ :, 1:, :]
        back_x = back_x[:, : -1 , :]


        back_images =  Variable(back_x).cuda()
        back_labels = Variable(back_y).long().cuda()


        x = torch.from_numpy(x)
        x = x.view(x.size()[0], x.size()[1], input_size)
        #y = torch.cat(( x[:, 1:, :], torch.zeros([x.size()[0], 1, input_size])), 1)
        y = x[:, 1:, :]
        x = x[:, :-1, :]

        
        images = Variable(x).cuda()
        labels = Variable(y).long().cuda()


        opt.zero_grad()
        outputs, states = rnn(images)
        back_outputs, back_states = back_rnn(back_images)
        shp = outputs.size()
        outputs_reshp = outputs.view([shp[0] * shp[1], num_classes])
        labels_reshp = labels.view(shp[0] * shp[1])
        back_outputs_reshp = back_outputs.view([shp[0] * shp[1], num_classes])
        back_labels_reshp = back_labels.view(shp[0] * shp[1])
        

        loss = criterion(outputs_reshp, labels_reshp)
        back_loss = criterion(back_outputs_reshp, back_labels_reshp)
    
        idx = [i for i in range(back_states.size()[1] - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        idx = Variable(idx).cuda()
        invert_backstates = back_states.index_select(1, idx)
        invert_backstates = invert_backstates.detach()

        states = states[:, 1:, : ]
        invert_backstates = invert_backstates [:, :-1, :]

        l2_loss = ((invert_backstates -  states) ** 2).mean()

        all_loss = loss + back_loss + 2.0 * l2_loss
        all_loss.backward()
        opt.step()

        t += time.time()
        
        if (step+1) % 10 == 0:
            log_line = 'Epoch [%d/%d], Step %d, all Loss: %f,  Loss: %f,  l2 Loss: %f, back Loss: %f, batch_time: %f \n' %(epoch, num_epochs, step+1, 784 * all_loss.data[0], 784 * loss.data[0], l2_loss.data[0], 784 * back_loss.data[0], t)
            print (log_line)
            with open(file_name, 'a') as f:
                f.write(log_line)


        if (step + 1) % 100 == 0:
            evaluate_valid(valid_x)

        step += 1
    
    if avg_valid_loss < hist_valid_loss:
        hist_valid_loss = avg_valid_loss
        save_param(rnn, model_file_name)

    # evaluate per epoch
    print '--- Epoch finished ----'
    evaluate_valid(valid_x)






























