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
        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        outputs = []
        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], input_t.size()[-1])
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            outputs += [c_t]
        outputs = torch.stack(outputs, 1).squeeze(2)
        shp=(outputs.size()[0], outputs.size()[1])
        out = outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        out = self.fc(out)
        out = out.view(shp[0], shp[1], self.num_classes)

        return out



class TWINNET_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TWINNET_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm3 = nn.LSTMCell(input_size, hidden_size)
        self.lstm4 = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.back_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        forward_outputs = []
        back_outputs = []

        h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        h_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        c_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())

        back_h_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        back_c_t = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        back_h_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())
        back_c_t2 = Variable(torch.zeros(x.size(0), self.hidden_size).cuda())


        for i, input_t in enumerate(x.chunk(x.size(1), dim=1)):
            input_t = input_t.contiguous().view(input_t.size()[0], 1)
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(c_t, (h_t2, c_t2))
            forward_outputs += [c_t2]
       
        for back_i, back_input_t in reversed(list(enumerate(x.chunk(x.size(1), dim=1)))):
            back_input_t = back_input_t.contiguous().view(back_input_t.size()[0], 1)
            back_h_t, back_c_t = self.lstm3(back_input_t, (back_h_t, back_c_t))
            back_h_t2, back_c_t2 = self.lstm4(back_c_t, (back_h_t2, back_c_t2))
            back_outputs += [back_c_t2]

        forward_outputs = torch.stack(forward_outputs, 1).squeeze(2)
        back_outputs = torch.stack(back_outputs, 1).squeeze(2)
        shp= (forward_outputs.size()[0], forward_outputs.size()[1])
        forward_out = forward_outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        forward_out = self.fc(forward_out)
        forward_out = forward_out.view(shp[0], shp[1], self.num_classes)
        
        back_out = back_outputs.contiguous().view(shp[0] *shp[1] , self.hidden_size)
        back_out = self.back_fc(back_out)
        back_out = back_out.view(shp[0], shp[1], self.num_classes)


        return [forward_out, back_out]


