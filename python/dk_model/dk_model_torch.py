import torch


class LinearDKModel:
    def __init__(self):
        super(LinearDKModel, self).__init__()
        # sampling period
        self.T = None
        # state space variables
        self.A = None
        self.B = None
        self.D = None
        self.E = None
        # state and input
        self.x = None
        self.u = None
        # use variable resistors
        self.vres = False
        # components
        self.components = None
        self.components_count = None

        self.Ao = None
        self.Bo = None
        self.Do = None
        self.Eo = None

        self.Z = None
        self.Rv = None
        self.Q = None
        self.Ux = None
        self.Uo = None
        self.Uu = None

        self.Nr = None
        self.Nv = None
        self.Gr = None
        self.Gx = None
        self.Nx = None
        self.Nu = None
        self.No = None

    def process(self, in_samples):
        out = torch.zeros([in_samples.shape[0], in_samples.shape[1]])  # output signal
        for ch in range(in_samples.shape[1]):  # iterate through audio channels
            for sample in range(in_samples.shape[0]):  # iterate through channel samples
                s = in_samples[sample, ch]
                self.load_input(s)
                if self.x.size == 0:
                    out_sample = self.E @ self.u
                else:
                    out_sample = self.D @ self.x[:, ch] + self.E @ self.u
                    self.x[:, ch:ch + 1] = self.A @ self.x[:, ch:ch + 1] + self.B @ self.u
                out[sample, ch] = out_sample
        return out

    # this method creates the state space variables according to Holters and Zölzer
    # (https://www.dafx.de/paper-archive/2011/Papers/21_e.pdf)
    def create_state_space(self):
        n_voltage_sources = self.Nu.shape[0]
        n_real_nodes = self.Nu.shape[1]

        # system matrix So computation
        if self.Nx.size == 0:
            So = torch.cat([self.Nr.T @ self.Gr @ self.Nr, self.Nu.T], dim=1)
            So = torch.cat([So, torch.cat([self.Nu, torch.zeros([n_voltage_sources, n_voltage_sources])], dim=1)], dim=0)
        # if there are capacitors
        else:
            So = torch.cat([self.Nr.T @ self.Gr @ self.Nr + self.Nx.T @ self.Gx @ self.Nx, self.Nu.T], dim=1)
            So = torch.cat([So, torch.cat([self.Nu, torch.zeros([n_voltage_sources, n_voltage_sources])], dim=1)], dim=0)
        # precomputable matrices
        So_inverse = torch.linalg.inv(So)

        Nvp = torch.cat([self.Nv, torch.zeros([self.Nv.shape[0], n_voltage_sources])], dim=1)
        Nxp = torch.cat([self.Nx, torch.zeros([self.Nx.shape[0], n_voltage_sources])], dim=1)
        Nop = torch.cat([self.No, torch.zeros([self.No.shape[0], n_voltage_sources])], dim=1)
        Nup = torch.cat([torch.zeros([n_real_nodes, n_voltage_sources]), torch.eye(n_voltage_sources)], dim=0)

        self.Q = Nvp @ So_inverse @ Nvp.T
        self.Ux = Nxp @ So_inverse @ Nvp.T
        self.Uo = Nop @ So_inverse @ Nvp.T
        self.Uu = Nup.T @ So_inverse @ Nvp.T

        self.Ao = 2 * self.Z @ self.Gx @ Nxp @ So_inverse @ Nxp.T - self.Z
        self.Bo = 2 * self.Z @ self.Gx @ Nxp @ So_inverse @ Nup
        self.Do = Nop @ So_inverse @ Nxp.T
        self.Eo = Nop @ So_inverse @ Nup

        # computation of the state space variables
        self.A = self.Ao - 2 * self.Z @ self.Gx @ self.Ux @ torch.linalg.inv(self.Rv + self.Q) @ self.Ux.T
        self.B = self.Bo - 2 * self.Z @ self.Gx @ self.Ux @ torch.linalg.inv(self.Rv + self.Q) @ self.Uu.T
        self.D = self.Do - self.Uo @ torch.linalg.inv(self.Rv + self.Q) @ self.Ux.T
        self.E = self.Eo - self.Uo @ torch.linalg.inv(self.Rv + self.Q) @ self.Uu.T

    def simple_state_space(self):
        n_voltage_sources = self.Nu.shape[0]
        n_real_nodes = self.Nu.shape[1]

        # system matrix S computation

        S = torch.cat([self.Nr.T @ self.Gr @ self.Nr + self.Nx.T @ self.Gx @ self.Nx, self.Nu.T], dim=1)
        S = torch.cat([S, torch.cat([self.Nu, torch.zeros([n_voltage_sources, n_voltage_sources])], dim=1)], dim=0)

        Nxp = torch.cat([self.Nx, torch.zeros([self.Nx.shape[0], n_voltage_sources])], dim=1)
        Nop = torch.cat([self.No, torch.zeros([self.No.shape[0], n_voltage_sources])], dim=1)
        Nup = torch.cat([torch.zeros([n_real_nodes, n_voltage_sources]), torch.eye(n_voltage_sources)], dim=0)

        Si = torch.linalg.inv(S)
        self.A = 2 * self.Z @ self.Gx @ Nxp @ Si @ Nxp.T - self.Z
        self.B = 2 * self.Z @ self.Gx @ Nxp @ Si @ Nup
        self.D = Nop @ Si @ Nxp.T
        self.E = Nop @ Si @ Nup

    def compute_steady_state(self):
        N = len(self.A)
        self.x = torch.linalg.solve((torch.eye(N) - self.A), (self.B @ self.u))

    def load_input(self, in_sample):
        self.u[0] = in_sample

    def build_model(self, ret=False, device=None):
        # allocate the matrices
        components_count = self.components_count
        components = self.components

        self.Nr = torch.zeros([components_count['n_res'], components_count['n_nodes']])  # resistors
        self.Gr = torch.zeros([components_count['n_res'], components_count['n_res']])

        if self.vres:
            self.Nv = torch.zeros([components_count['n_vres'], components_count['n_nodes']])  # variable resistors
            self.Rv = torch.zeros([components_count['n_vres'], components_count['n_vres']])
        else:
            self.Nv = []

        self.Nx = torch.zeros([components_count['n_caps'], components_count['n_nodes']])  # capacitors
        self.Gx = torch.zeros([components_count['n_caps'], components_count['n_caps']])
        self.Z = torch.zeros([components_count['n_caps'], components_count['n_caps']])

        self.Nu = torch.zeros([components_count['n_inputs'], components_count['n_nodes']])  # inputs
        self.No = torch.zeros([components_count['n_outputs'], components_count['n_nodes']])  # outputs
        # allocate the input vector
        self.u = torch.zeros([components_count['n_inputs'], 1])
        # component count helper variables
        n_res, n_vres, n_caps, n_inputs, n_outputs = 0, 0, 0, 0, 0
        # populate the matrices
        for component in components:
            if component.type == 'res':  # resistor
                if component.nodes[0, 0] > 0:
                    self.Nr[n_res, component.nodes[0, 0] - 1] = 1
                if component.nodes[0, 1] > 0:
                    self.Nr[n_res, component.nodes[0, 1] - 1] = -1
                self.Gr[n_res, n_res] = 1 / component.value
                n_res += 1
            elif component.type == 'vres' and self.vres:  # variable resistor
                if component.nodes[0, 0] > 0:
                    self.Nv[n_vres, component.nodes[0, 0] - 1] = 1
                if component.nodes[0, 1] > 0:
                    self.Nv[n_vres, component.nodes[0, 1] - 1] = -1
                self.Rv[n_vres, n_vres] = component.value
                n_vres += 1
            elif component.type == 'cap':  # capacitor
                if component.nodes[0, 0] > 0:
                    self.Nx[n_caps, component.nodes[0, 0] - 1] = 1
                if component.nodes[0, 1] > 0:
                    self.Nx[n_caps, component.nodes[0, 1] - 1] = -1
                self.Z[n_caps, n_caps] = 1
                self.Gx[n_caps, n_caps] = 2 * component.value / self.T
                n_caps += 1
            elif component.type == 'in':  # input port
                if component.nodes[0, 0] > 0:
                    self.Nu[n_inputs, component.nodes[0, 0] - 1] = 1
                if component.nodes[0, 1] > 0:
                    self.Nu[n_inputs, component.nodes[0, 1] - 1] = -1
                self.u[n_inputs, 0] = component.value
                n_inputs += 1
            elif component.type == 'out':  # output port
                if component.nodes[0, 0] > 0:
                    self.No[n_outputs, component.nodes[0, 0] - 1] = 1
                if component.nodes[0, 1] > 0:
                    self.No[n_outputs, component.nodes[0, 1] - 1] = -1
                n_outputs += 1
        if self.vres:
            self.create_state_space()
        else:
            self.simple_state_space()
        self.compute_steady_state()
        if ret:
            return [self.Nr.to(device), self.Gr.to(device), self.Nx.to(device), self.Gx.to(device),
                    self.Z.to(device), self.Nu.to(device), self.No.to(device)]
