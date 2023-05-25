import torch
import torch.nn as nn
import pytorch_lightning as pl

from dk_model.tone_stack_torch import JVMToneStack
from models.cond_blocks import CondBlock


class ToneStack(pl.LightningModule):
    def __init__(
            self,
            sr,
            state_size,
            n_targets,
            cond_input='labels',
            cond_process='pot',
            freeze_cond_block=False
            ):
        super(ToneStack, self).__init__()

        # labels or one hot encoded vectors
        self.cond_input = cond_input
        # process cond input either by mlps or by learnable potentiometer tapers
        self.cond_process = cond_process
        # size of the one hot encoded vector (number of target sounds)
        self.n_targets = n_targets
        #
        self.freeze_cond_block = freeze_cond_block
        self.sr = sr

        self.state_size = state_size
        self.state = None
        self.recurse_state = None

        self.ts = JVMToneStack(sr=self.sr)
        # matrices for state space computation
        self.A, self.B, self.D, self.E = None, None, None, None
        # resistors and caps
        self.Nr, self.Gr, self.Nx, self.Gx = None, None, None, None
        # variable resistors
        self.Nv, self.Rv, self.Q, self.RvQ = None, None, None, None
        # rest of the matrices
        self.Nu, self.No, self.Z = None, None, None
        self.Nvp, self.Nxp, self.Nop, self.Nup = None, None, None, None

        # frequency sampling vector
        self.z = None

        # initial component values
        self.RT = 200000
        self.RM = 20000
        self.RB = 1000000
        self.RV = 1000000

        self.R1 = 33000
        self.R2 = 39000

        self.C1 = 470e-12
        self.C2 = 22e-9
        self.C3 = 22e-9

        # trainable scaling coeffs for the virtual components
        self.alpha_rt = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.alpha_rm = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.alpha_rb = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        self.alpha_r1 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.alpha_r2 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        self.alpha_c1 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.alpha_c2 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.alpha_c3 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        # conditioning block
        self.cond_block = CondBlock(n_targets=self.n_targets,
                                    cond_input=self.cond_input,
                                    cond_process=self.cond_process)
        if self.freeze_cond_block:
            self.cond_block.freeze()
        self.finetuning = False

    def init(self, new_batch_size):
        # put sampling vector to GPU
        self.z = torch.exp(
            torch.complex(torch.zeros(1, dtype=torch.double), -torch.ones(1, dtype=torch.double)) *
            torch.linspace(0, torch.pi, 1 + self.state_size // 2))
        self.z = (self.z.to(self.device)).unsqueeze(1).repeat(1, new_batch_size)
        # get dk method matrices
        matrices = self.ts.get_matrices(self.device)
        self.Nr = matrices[0]
        self.Gr = matrices[1]
        self.Nx = matrices[2]
        self.Gx = matrices[3]
        self.Nv = matrices[4]
        self.Rv = matrices[5].unsqueeze(0).repeat(new_batch_size, 1, 1)
        self.Nu = matrices[6]
        self.No = matrices[7]
        self.Z = matrices[8]
        self.Q = matrices[9]

        n_vsrcs = self.Nu.shape[0]
        n_nodes = self.Nu.shape[1]

        self.Nvp = torch.cat([self.Nv, torch.zeros([self.Nv.shape[0], n_vsrcs], device=self.device)], dim=1)
        self.Nxp = torch.cat([self.Nx, torch.zeros([self.Nx.shape[0], n_vsrcs], device=self.device)], dim=1)
        self.Nop = torch.cat([self.No, torch.zeros([self.No.shape[0], n_vsrcs], device=self.device)], dim=1)
        self.Nup = torch.cat([torch.zeros([n_nodes, n_vsrcs], device=self.device),
                              torch.eye(n_vsrcs, device=self.device)], dim=0)

    def resize_tensors(self, new_batch_size):
        # put sampling vector to GPU
        self.z = torch.exp(
            torch.complex(torch.zeros(1, dtype=torch.double), -torch.ones(1, dtype=torch.double)) *
            torch.linspace(0, torch.pi, 1 + self.state_size // 2))
        # adjust sampling vector and Rv tensor according to the batch size
        self.z = (self.z.to(self.device)).unsqueeze(1).repeat(1, new_batch_size)
        matrices = self.ts.get_matrices(self.device)
        self.Rv = matrices[5].unsqueeze(0).repeat(new_batch_size, 1, 1)

    def get_component_values(self):
        alpha_rb = (0.8 + torch.sigmoid(self.alpha_rb) * 0.4)
        alpha_rm = (0.8 + torch.sigmoid(self.alpha_rm) * 0.4)
        alpha_rt = (0.8 + torch.sigmoid(self.alpha_rt) * 0.4)
        alpha_r1 = (0.8 + torch.sigmoid(self.alpha_r1) * 0.4)
        alpha_r2 = (0.8 + torch.sigmoid(self.alpha_r2) * 0.4)

        alpha_c1 = (0.8 + torch.sigmoid(self.alpha_c1) * 0.4)
        alpha_c2 = (0.8 + torch.sigmoid(self.alpha_c2) * 0.4)
        alpha_c3 = (0.8 + torch.sigmoid(self.alpha_c3) * 0.4)

        RB = alpha_rb * self.RB
        RM = alpha_rm * self.RM
        RT = alpha_rt * self.RT
        R1 = alpha_r1 * self.R1
        R2 = alpha_r2 * self.R2

        C1 = alpha_c1 * self.C2
        C2 = alpha_c2 * self.C2
        C3 = alpha_c3 * self.C3

        c_vals = {
            'caps_alphas': {
                'alpha_c1': alpha_c1,
                'alpha_c2': alpha_c2,
                'alpha_c3': alpha_c3},
            'caps': {
                'C1': C1,
                'C2': C2,
                'C3': C3},
            'res_alphas': {
                'alpha_r1': alpha_r1,
                'alpha_r2': alpha_r2,
                'alpha_rb': alpha_rb,
                'alpha_rm': alpha_rm,
                'alpha_rt': alpha_rt},
            'res': {
                'R1': R1,
                'R2': R2,
                'RB': RB,
                'RM': RM,
                'RT': RT}}
        return c_vals

    def forward(self, x, cond):
        if self.finetuning:
            processed_cond = cond
        else:
            processed_cond = self.cond_block(cond)
        out = self.freq_filt(x, processed_cond)
        return out

    def freq_filt(self, x, processed_cond):
        self.update_components(processed_cond)
        self.update_state_space()

        self.state = torch.cat((self.state[:, x.shape[1]:, :], x), dim=1)
        state_fft = torch.fft.rfft(self.state.to(torch.double), dim=1)

        h = self.h_from_state_space()

        out_fft = torch.mul(state_fft, h)
        out = (torch.fft.irfft(out_fft, dim=1)).to(torch.float32)
        return out[:, -x.shape[1]:, :]

    def state_space_filt(self, u, processed_cond):
        self.update_components(processed_cond)
        self.update_state_space()

        filt_out = torch.zeros(u.shape, device=self.device)
        for i in range(u.shape[1]):
            filt_out[:, i: i + 1, :] = self.D @ self.recurse_state + self.E @ u[:, i: i + 1, :]
            self.recurse_state = self.A @ self.recurse_state + self.B @ u[:, i: i + 1, :]
        return filt_out

    def update_components(self, processed_cond):
        RB = (0.8 + torch.sigmoid(self.alpha_rb) * 0.4) * self.RB
        RM = (0.8 + torch.sigmoid(self.alpha_rm) * 0.4) * self.RM
        RT = (0.8 + torch.sigmoid(self.alpha_rt) * 0.4) * self.RT
        self.update_pots(processed_cond, RB, RM, RT)
        self.update_resistors(RM)
        self.update_caps()

    def update_pots(self, processed_cond, RB, RM, RT):
        # pot values are different for each cond
        # Rv has a shape of (batch_size, num_vres, num_vres)
        Rv = torch.zeros_like(self.Rv, device=self.device)
        # TREBLE
        Rv[:, 0, 0] = (1 - processed_cond[2]) * RT
        Rv[:, 1, 1] = processed_cond[2] * RT
        # MID
        Rv[:, 2, 2] = (2 * (1 - processed_cond[1])) / (2 - (1 - processed_cond[1])) * RM
        Rv[:, 3, 3] = (2 * processed_cond[1]) / (2 - processed_cond[1]) * RM
        # BASS
        Rv[:, 4, 4] = processed_cond[0] * RB
        # VOLUME
        Rv[:, 5, 5] = torch.tensor([0 * self.RV])
        Rv[:, 6, 6] = torch.tensor([2 * self.RV])
        self.RvQ = torch.linalg.inv(Rv + self.Q)

    def update_resistors(self, RM):
        # resistors are the same for all cond values
        Gr = torch.zeros_like(self.Gr, device=self.device)
        # 1% tol
        R1 = (0.8 + torch.sigmoid(self.alpha_r1) * 0.4) * self.R1
        R2 = (0.8 + torch.sigmoid(self.alpha_r2) * 0.4) * self.R2
        Gr[0, 0] = 1 / R1
        # mid parallel resistors
        Gr[1, 1] = 1 / (RM * 2)
        Gr[2, 2] = 1 / (RM * 2)
        Gr[3, 3] = 1 / R2
        # vol parallel resistors
        Gr[4, 4] = 1 / (self.RV * 2)
        Gr[5, 5] = 1 / (self.RV * 2)
        self.Gr = Gr

    def update_caps(self):
        # capacitors are the same for all cond values
        Gx = torch.zeros_like(self.Gx, device=self.device)
        # 10% tol
        C1 = (0.8 + torch.sigmoid(self.alpha_c1) * 0.4) * self.C1
        C2 = (0.8 + torch.sigmoid(self.alpha_c2) * 0.4) * self.C2
        C3 = (0.8 + torch.sigmoid(self.alpha_c3) * 0.4) * self.C3
        Gx[0, 0] = 2 * C1 / self.ts.T
        Gx[1, 1] = 2 * C2 / self.ts.T
        Gx[2, 2] = 2 * C3 / self.ts.T
        self.Gx = Gx

    def update_state_space(self):
        n_vsrc = self.Nu.shape[0]
        # system matrix So is computed only once, since the resistors and capacitors are the same for
        # all conditioning values
        So = torch.cat([self.Nr.T @ self.Gr @ self.Nr + self.Nx.T @ self.Gx @ self.Nx, self.Nu.T], dim=1)
        So = torch.cat([So, torch.cat([self.Nu, torch.zeros([n_vsrc, n_vsrc], device=self.device)], dim=1)], dim=0)
        So_inverse = torch.linalg.inv(So)

        # following matrices are the same as with single state space representation
        self.Q = self.Nvp @ So_inverse @ self.Nvp.T

        Ux = self.Nxp @ So_inverse @ self.Nvp.T
        Uo = self.Nop @ So_inverse @ self.Nvp.T
        Uu = self.Nup.T @ So_inverse @ self.Nvp.T
        ZGx = 2 * self.Z @ self.Gx

        Ao = ZGx @ self.Nxp @ So_inverse @ self.Nxp.T - self.Z
        Bo = ZGx @ self.Nxp @ So_inverse @ self.Nup
        Do = self.Nop @ So_inverse @ self.Nxp.T
        Eo = self.Nop @ So_inverse @ self.Nup

        # computation of state space matrices with added batch dimension
        # the batch dim is simply added by the fact the self.RvQ has it
        self.A = Ao - ZGx @ Ux @ self.RvQ @ Ux.T
        self.B = Bo - ZGx @ Ux @ self.RvQ @ Uu.T
        self.D = Do - Uo @ self.RvQ @ Ux.T
        self.E = Eo - Uo @ self.RvQ @ Uu.T

    def h_from_state_space(self):
        # transfer funcs polynomial coeffs
        den = self.batch_poly(self.A)
        num = self.batch_poly(self.A - (self.B @ self.D)) + (self.E.squeeze(-1).repeat(1, den.shape[1]) - 1.) * den
        # evaluate denominator and numerator polynomials
        a = self.batch_polyval(den)
        b = self.batch_polyval(num)
        # compute freq responses and add dims (batch_size, 1 + state_size // 2, 1)
        return (b / a).unsqueeze(-1)

    # implemented according to np.poly() with added batch dim
    def batch_poly(self, seq_of_zeros):
        # eigvals() func returns complex64 numbers, complex128 when the input is float64
        seq = torch.linalg.eigvals(seq_of_zeros)
        # conv1d() can produce rounding errors when using float32 as a dtype (default)
        # this can result in very inaccurate frequency responses due to badly computed coeffs
        # can be fixed by using complex64 or complex128 as a dtype for kernel and coeffs
        coeffs = torch.ones((1, seq.shape[0], 1), device=self.device, dtype=torch.complex128)
        for i in range(seq.shape[1]):
            kernel = torch.ones(seq.shape[0], 1, 2, device=self.device, dtype=torch.complex128)
            kernel[:, 0, 0] = -seq[:, i]
            coeffs = nn.functional.conv1d(coeffs, kernel, padding=1, groups=seq.shape[0], bias=None)
        # return only the real part since the imag part is zero anyway
        return coeffs.squeeze(0).real

    # implemented according to np.polyval() with added batch dim
    def batch_polyval(self, coeffs):
        c0 = coeffs[:, -1] + self.z * 0
        for i in range(2, coeffs.shape[1] + 1):
            c0 = coeffs[:, -i] + c0 * self.z
        return c0.permute(1, 0)

    def reset_state(self, batch_size):
        self.state = torch.zeros((batch_size, self.state_size, 1), device=self.device)
        self.recurse_state = torch.zeros((batch_size, 3, 1), device=self.device)

    def detach_state(self):
        self.state = self.state.clone().detach()
        self.recurse_state = self.recurse_state.clone().detach()
        self.Q = self.Q.clone().detach()
