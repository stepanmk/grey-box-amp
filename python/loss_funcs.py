import torch
import torch.nn as nn
# import torch.nn.functional as F


# Error to signal loss
class ESRLoss(nn.Module):
    def __init__(self):
        super(ESRLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output, target):
        loss = torch.add(target, -output)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss


# log Error to signal loss
class LogESRLoss(nn.Module):
    def __init__(self):
        super(LogESRLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output, target):
        loss = torch.add(target, -output)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = 10 * torch.log10(torch.div(loss, energy) + self.epsilon)
        return loss


# DC loss
class DCLoss(nn.Module):
    def __init__(self):
        super(DCLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output, target):
        loss = torch.pow(torch.add(torch.mean(target, 0), -torch.mean(output, 0)), 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss


# Surface similarity parameter loss
class SSPLoss(nn.Module):
    def __init__(self):
        super(SSPLoss, self).__init__()
        self.epsilon = 0.00001

    def forward(self, output, target):
        # ffts
        output_fft = torch.fft.rfft(output, dim=1)
        target_fft = torch.fft.rfft(target, dim=1)
        # ssp
        num = torch.sqrt(torch.trapz(torch.pow(torch.abs(output_fft - target_fft), 2), dim=1))
        denom1 = torch.sqrt(torch.trapz(torch.pow(torch.abs(output_fft), 2), dim=1))
        denom2 = torch.sqrt(torch.trapz(torch.pow(torch.abs(target_fft), 2), dim=1))
        loss = torch.mean(torch.div(num, denom1 + denom2 + self.epsilon))
        return loss
