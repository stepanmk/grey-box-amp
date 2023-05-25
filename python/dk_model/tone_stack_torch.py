from dk_model.dk_model_torch import LinearDKModel
from dk_model.components_torch import *


class JVMToneStack(LinearDKModel):
    def __init__(self,
                 sr=44100,
                 alpha_t=0.5,
                 alpha_m=0.5,
                 alpha_b=0.5,
                 alpha_v=1.0):
        super().__init__()
        self.vres = True
        self.T = 1 / sr

        self.alpha_t1 = 0.
        self.alpha_t2 = 0.
        self.alpha_m1 = 0.
        self.alpha_m2 = 0.
        self.alpha_b = 0.
        self.alpha_v1 = 0.
        self.alpha_v2 = 0.

        self.recompute_alphas(alpha_t, alpha_m, alpha_b, alpha_v)

        self.RT = 200000
        self.RM = 20000
        self.RB = 1000000
        self.RV = 1000000

        self.components = [
            Resistor('R1', [1, 2], 33000),
            # TREBLE #
            VariableResistor('VR1_1', [3, 4], self.RT * self.alpha_t1),
            VariableResistor('VR1_2', [4, 5], self.RT * self.alpha_t2),
            # MID #
            VariableResistor('VR2_1', [6, 7], self.RM * self.alpha_m1),
            VariableResistor('VR2_2', [7, 0], self.RM * self.alpha_m2),
            # parallel constant resistors to avoid floating nodes
            Resistor('R2_1', [6, 7], self.RM * 2),
            Resistor('R2_2', [7, 0], self.RM * 2),
            # BASS #
            VariableResistor('VR3', [5, 6], self.RB * self.alpha_b),
            # VOLUME #
            Resistor('R5', [4, 8], 39000),
            VariableResistor('VR4_1', [8, 9], self.RV * self.alpha_v1),
            VariableResistor('VR4_2', [9, 0], self.RV * self.alpha_v2),
            # parallel constant resistors to avoid floating nodes
            Resistor('R4_1', [8, 9], self.RV * 2),
            Resistor('R4_2', [9, 0], self.RV * 2),

            # capacitors
            Capacitor('C1', [1, 3], 470e-12),
            Capacitor('C2', [2, 5], 22e-9),
            Capacitor('C3', [2, 7], 22e-9),
            # input & output ports
            InputPort('In', [1, 0], 0),
            OutputPort('Out', [9, 0])
        ]
        self.components_count = {
            'n_res': 6,
            'n_vres': 7,
            'n_caps': 3,
            'n_inputs': 1,
            'n_outputs': 1,
            'n_nodes': 9,
        }
        self.build_model()

    def recompute_alphas(self, t, m, b, v):
        self.alpha_t1 = 1 - t
        self.alpha_t2 = t
        self.alpha_m1 = (2 * (1 - m)) / (2 - (1 - m))
        self.alpha_m2 = (2 * m) / (2 - m)
        self.alpha_b = b
        self.alpha_v1 = (2 * (1 - v)) / (2 - (1 - v))
        self.alpha_v2 = (2 * v) / (2 - v)

    def get_matrices(self, device):
        return [self.Nr.to(device), self.Gr.to(device), self.Nx.to(device), self.Gx.to(device),
                self.Nv.to(device), self.Rv.to(device), self.Nu.to(device), self.No.to(device),
                self.Z.to(device), self.Q.to(device)]
