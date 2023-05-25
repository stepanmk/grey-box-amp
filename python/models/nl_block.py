# import torch
import torch.nn as nn
import pytorch_lightning as pl


class NLAmp(pl.LightningModule):
    def __init__(self, params):
        super(NLAmp, self).__init__()
        self.type = params.pop('type')
        self.params = params
        if self.type == 'GRU':
            self.model = GRUAmp(**self.params)
        if self.type == 'LSTM':
            self.model = LSTMAmp(**self.params)

    def forward(self, x):
        return self.model(x)

    def reset_state(self, batch_size):
        self.model.reset_state(batch_size)

    def detach_state(self):
        self.model.detach_state()


class GRUAmp(pl.LightningModule):
    def __init__(self, hidden_size, input_size, res):
        super(GRUAmp, self).__init__()
        self.rec = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Linear(in_features=hidden_size, out_features=1)
        self.state = None
        self.res = res

    def forward(self, x):
        res = x[:, :, 0:1]
        x, self.state = self.rec(x, self.state)
        x = self.lin(x)
        if self.res:
            x = x + res
        return x, self.state

    def reset_state(self, batch_size=None):
        self.state = None

    def detach_state(self):
        self.state = self.state.detach()


class LSTMAmp(pl.LightningModule):
    def __init__(self, hidden_size, input_size, res):
        super(LSTMAmp, self).__init__()
        self.rec = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.lin = nn.Linear(in_features=hidden_size, out_features=1)
        self.state = None
        self.res = res

    def forward(self, x):
        res = x[:, :, 0:1]
        x, self.state = self.rec(x, self.state)
        x = self.lin(x)
        if self.res:
            x = x + res
        return x, self.state

    def reset_state(self, batch_size=None):
        self.state = None

    def detach_state(self):
        state = list(self.state)
        state[0] = state[0].detach()
        state[1] = state[1].detach()
        self.state = tuple(state)
