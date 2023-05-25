import torch


class Resistor:
    def __init__(self, name, nodes, value):
        self.nodes = torch.zeros([2, 2], dtype=torch.int)
        self.nodes[0, :] = torch.IntTensor(nodes)
        self.name = name
        self.value = value
        self.type = 'res'


class VariableResistor:
    def __init__(self, name, nodes, value):
        self.nodes = torch.zeros([2, 2], dtype=torch.int)
        self.nodes[0, :] = torch.IntTensor(nodes)
        self.name = name
        self.value = value
        self.type = 'vres'


class Capacitor:
    def __init__(self, name, nodes, value):
        self.nodes = torch.zeros([2, 2], dtype=torch.int)
        self.nodes[0, :] = torch.IntTensor(nodes)
        self.name = name
        self.value = value
        self.type = 'cap'


class InputPort:
    def __init__(self, name, nodes, value):
        self.nodes = torch.zeros([2, 2], dtype=torch.int)
        self.nodes[0, :] = torch.IntTensor(nodes)
        self.name = name
        self.value = value
        self.type = 'in'


class OutputPort:
    def __init__(self, name, nodes):
        self.nodes = torch.zeros([2, 2], dtype=torch.int)
        self.nodes[0, :] = torch.IntTensor(nodes)
        self.name = name
        self.value = 0
        self.type = 'out'
