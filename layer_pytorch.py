import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets
import torch.nn.functional as F
from torch.nn.parameter import Parameter



class RNN_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        h_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())

        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], 1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            outputs += [c_t2]
        outputs = torch.stack(outputs, 1).squeeze(2)
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out




