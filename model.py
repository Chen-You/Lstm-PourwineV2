import math

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torchsnooper

from config import params


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.num_layers = params['num_layers']
        self.batch_size = params['batch_size']
        self.hidden_size = params['hidden_size']

        self.gru = nn.GRU(
            input_size=params['previous_state']*params['input_size'],
            hidden_size=params['hidden_size'],
            num_layers=params['num_layers'],
            batch_first=params['batch_first'],
            bidirectional=params['bidirectional'],
            dropout=params['dropout'],
        )

        self.out = nn.Linear(params['hidden_size'], params['output_features'])

    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden) # output.size(): B*T*F
        output = output[:, -1:, :].contiguous() # output.size(): B*1*F
        output = self.out(output)
        return output, hidden

    def initHidden(self):
        # **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`
        hidden = Variable(torch.empty(self.num_layers, self.batch_size, self.hidden_size))
        init.orthogonal_(hidden)
        return hidden

def count_parameters(model):

    for p in model.parameters():
        print(p.size())
        print(p.numel())

    # for name, param in model.named_parameters():
        # print(name, param


    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    model = Model(params)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    # print()
    # print(model)
    print(count_parameters(model))

