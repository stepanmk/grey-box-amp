import torch
import pytorch_lightning as pl

import loss_funcs
from models.nl_block import NLAmp
from models.ts_block import ToneStack
from matplotlib import pyplot as plt
from utils.ir_funcs import get_ir, get_freq_resp
import numpy as np

import os
import soundfile as sf
import auraloss
import wandb
import pickle


class AmpModel(pl.LightningModule):
    def __init__(self,
                 train_lf,
                 model_name,
                 n_targets,
                 log_subfolder,
                 pre_type='LSTM',
                 pre_hidden=40,
                 pre_res=True,
                 include_poweramp=False,
                 poweramp_type='GRU',
                 poweramp_hidden=8,
                 poweramp_res=True,
                 trunc_steps=2048,
                 warmup=1000,
                 l_rate=0.002,
                 l_rate_ts=0.002,
                 batch_size=80,
                 batch_size_val=12,
                 cond_input='labels',
                 cond_process='pot',
                 rnn_only=False,
                 freeze_cond_block=False,
                 gain_cond=False,
                 sr=44100
                 ):
        super(AmpModel, self).__init__()
        self.l_rate = l_rate
        self.l_rate_ts = l_rate_ts
        self.truncated_bptt_steps = trunc_steps
        self.warmup = warmup

        self.batch_size = batch_size
        self.batch_size_val = batch_size_val
        self.model_name = model_name
        self.cond_input = cond_input
        self.cond_process = cond_process
        self.n_targets = n_targets
        self.log_subfolder = log_subfolder

        self.rnn_only = rnn_only
        self.include_poweramp = include_poweramp
        self.freeze_cond_block = freeze_cond_block
        self.gain_cond = gain_cond
        self.sr = sr

        self.pre_type = pre_type
        self.pre_hidden = pre_hidden
        self.pre_res = pre_res
        self.preamp = NLAmp({'type': self.pre_type,
                             'hidden_size': self.pre_hidden,
                             'input_size': 4 if self.rnn_only else 1,
                             'res': self.pre_res})
        self.tone_stack = ToneStack(sr=self.sr,
                                    state_size=self.truncated_bptt_steps * 2,
                                    n_targets=self.n_targets,
                                    cond_input=self.cond_input,
                                    cond_process=self.cond_process,
                                    freeze_cond_block=self.freeze_cond_block)
        self.poweramp_type = poweramp_type
        self.poweramp_hidden = poweramp_hidden
        self.poweramp_res = poweramp_res
        if self.include_poweramp:
            self.poweramp = NLAmp({'type': self.poweramp_type,
                                   'hidden_size': self.poweramp_hidden,
                                   'input_size': 1,
                                   'res': self.poweramp_res})

        self.loss_functions = {
            'ESR': loss_funcs.ESRLoss(),
            'LogESR': loss_funcs.LogESRLoss(),
            'SSP': loss_funcs.SSPLoss(),
            'MSE': torch.nn.MSELoss(),
            'MAE': torch.nn.L1Loss(),
            # order of the dims edited directly in the auraloss code
            'STFT': auraloss.freq.STFTLoss()
        }

        self.lf = train_lf
        self.loss_fn = self.loss_functions[self.lf]
        self.loss_functions.pop(self.lf)

        self.val_selected = [0, 4, 6, 8]
        self.test_selected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.val_inputs = []
        self.val_targets = []
        self.val_outputs = []

        self.test_inputs = []
        self.test_targets = []
        self.test_outputs = []

        self.final_val = False
        self.sweeps_eval = False

        self.metrics_dict = {}
        self.save_hyperparameters()

    def configure_optimizers(self, l_rate=0.002):
        if self.rnn_only:
            trainable_params = [{'params': self.preamp.parameters(), 'lr': self.l_rate}]
        else:
            trainable_params = [{'params': self.preamp.parameters(), 'lr': self.l_rate},
                                {'params': self.tone_stack.parameters(), 'lr': self.l_rate_ts}]
        if self.include_poweramp:
            trainable_params.append({'params': self.poweramp.parameters(), 'lr': self.l_rate})
        optim = torch.optim.Adam(trainable_params)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optim, factor=0.5, patience=5, verbose=True)
        return {'optimizer': optim,
                'lr_scheduler': {'scheduler': lr_scheduler,
                                 'monitor': f'{self.lf}_val',
                                 'frequency': 2}}

    def reset_states(self, batch_size):
        self.preamp.reset_state(batch_size)
        self.tone_stack.reset_state(batch_size)
        if self.include_poweramp:
            self.poweramp.reset_state(batch_size)

    def detach_states(self):
        self.preamp.detach_state()
        self.tone_stack.detach_state()
        if self.include_poweramp:
            self.poweramp.detach_state()

    def on_train_start(self):
        self.tone_stack.init(self.batch_size)

    def on_train_epoch_start(self):
        self.tone_stack.resize_tensors(self.batch_size)

    def on_validation_epoch_start(self):
        self.tone_stack.resize_tensors(self.batch_size_val)

    def on_test_epoch_start(self):
        self.tone_stack.init(self.batch_size_val)
        self.tone_stack.resize_tensors(self.batch_size_val)

    def on_validation_end(self):
        self.per_cond_losses(stage='val')
        if not self.rnn_only:
            wandb.log({'components': self.tone_stack.get_component_values()})
        if self.final_val and not self.rnn_only:
            self.plot_pot_tapers()
            wandb.log({'components_best': self.tone_stack.get_component_values()})

    def on_test_epoch_end(self):
        if self.sweeps_eval:
            self.plot_mag_responses()
        else:
            self.per_cond_losses(stage='test')

    def tbptt_split_batch(self, batch, split_size):
        total_steps = batch[0].shape[1]
        splits = [[x[:, :self.warmup, :] for i, x in enumerate(batch[0:2])]]
        splits[0].append(batch[2])
        for t in range(self.warmup, total_steps, split_size):
            batch_split = [x[:, t: t + split_size, :] for i, x in enumerate(batch[0:2])]
            batch_split.append(batch[2])
            splits.append(batch_split)
        return splits

    @staticmethod
    def val_split_batch(batch, split_size):
        total_steps = batch[0].shape[1]
        splits = []
        for t in range(0, total_steps, split_size):
            batch_split = [x[:, t: t + split_size, :] for i, x in enumerate(batch[0:2])]
            batch_split.append(batch[2])
            splits.append(batch_split)
        return splits

    def forward(self, x, cond):
        if self.rnn_only:
            cond = cond.unsqueeze(1).repeat(1, x.shape[1], 1)
            x = torch.cat([x, cond], dim=-1)
            y_hat, hiddens = self.preamp(x)
        else:
            y_preamp, hiddens = self.preamp(x)
            y_hat = self.tone_stack(y_preamp, cond)
            if self.include_poweramp:
                y_hat, hiddens = self.poweramp(y_hat)
        return y_hat, hiddens

    def val_forward(self, x, cond):
        if self.rnn_only:
            cond = cond.unsqueeze(1).repeat(1, x.shape[1], 1)
            x = torch.cat([x, cond], dim=-1)
            y_hat, _ = self.preamp(x)
        else:
            y_preamp, _ = self.preamp(x)
            y_hat = self.tone_stack.state_space_filt(y_preamp, cond)
            if self.include_poweramp:
                y_hat, _ = self.poweramp(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx, hiddens=None):
        warmup_step = hiddens is None
        x, y, cond = batch
        if warmup_step:
            self.reset_states(batch[0].shape[0])
            y_hat, hiddens = self(x, cond)
            loss = torch.zeros(1, device=self.device, requires_grad=True)
            return {'loss': loss, 'hiddens': hiddens}
        else:
            # detach states so that the backprop is computed only over trunc_steps
            self.detach_states()
            y_hat, hiddens = self(x, cond)
            loss = self.loss_fn(y_hat, y)
            self.log(f'{self.lf}_train', loss, on_epoch=True, on_step=False)
            return {'loss': loss, 'hiddens': hiddens}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            self.reset_states(batch[0].shape[0])
            splits = self.val_split_batch(batch=batch, split_size=self.truncated_bptt_steps)
            y_hats = []
            for split in splits:
                x, y, cond = split
                y_hat, _ = self(x, cond)
                y_hats.append(y_hat)
            y_hat_cat = torch.cat(y_hats, dim=1)
            x, y, cond = batch
            val_loss = self.loss_fn(y_hat_cat, y)
            self.val_inputs.append(x)
            self.val_targets.append(y)
            self.val_outputs.append(y_hat_cat)
            if self.final_val:
                self.log(f'{self.lf}_val_best', val_loss, on_epoch=True, on_step=False)
                self.calc_vlosses(y_hat_cat, y, 'val_best')
            else:
                self.log(f'{self.lf}_val', val_loss, on_epoch=True, on_step=False)
                self.calc_vlosses(y_hat_cat, y, 'val')

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            self.reset_states(batch[0].shape[0])
            splits = self.val_split_batch(batch=batch, split_size=self.truncated_bptt_steps)
            y_hats = []
            for split in splits:
                x, y, cond = split
                y_hat, _ = self(x, cond)
                # y_hat = self.val_forward(x, cond)
                y_hats.append(y_hat)
            y_hat_cat = torch.cat(y_hats, dim=1)
            x, y, cond = batch
            test_loss = self.loss_fn(y_hat_cat, y)
            self.test_inputs.append(x)
            self.test_targets.append(y)
            self.test_outputs.append(y_hat_cat)
            set_type = self.trainer.test_dataloaders[0].dataset.set_type
            if not self.sweeps_eval:
                self.log(f'{self.lf}_{set_type}',
                         test_loss, on_epoch=True, on_step=False)
                # rest of the test losses
                self.calc_vlosses(y_hat_cat, y, set_type)

    def calc_vlosses(self, y_hat, y, stage, log=True):
        for func in self.loss_functions.keys():
            loss = self.loss_functions[func](y_hat, y)
            if log:
                self.log(f'{func}_{stage}', loss, on_epoch=True, on_step=False)
            else:
                self.metrics_dict[f'{func}_{stage}'] = float(loss.item())

    def per_cond_losses(self, stage):
        if stage == 'val':
            inputs = self.val_inputs
            targets = self.val_targets
            outputs = self.val_outputs
            split_size = self.trainer.val_dataloaders[0].dataset.segs_per_cond
            settings_names = self.trainer.val_dataloaders[0].dataset.settings_names
            set_type = self.trainer.val_dataloaders[0].dataset.set_type
        else:
            inputs = self.test_inputs
            targets = self.test_targets
            outputs = self.test_outputs
            split_size = self.trainer.test_dataloaders[0].dataset.segs_per_cond
            settings_names = self.trainer.test_dataloaders[0].dataset.settings_names
            set_type = self.trainer.test_dataloaders[0].dataset.set_type

        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)
        outputs = torch.cat(outputs, dim=0)

        c = 0
        loss_dict = {}
        loss_dict_save = {}
        loss_dict_spectral = {}
        loss_dict_spectral_save = {}
        total_steps = targets.shape[0]
        for t in range(0, total_steps, split_size):
            cond_output = outputs[t: t + split_size, :, :]
            cond_target = targets[t: t + split_size, :, :]
            per_cond_loss = self.loss_fn(cond_output, cond_target)
            per_cond_loss_spectral = self.loss_functions['STFT'](cond_output, cond_target)
            settings_name = settings_names[c]
            loss_dict[settings_name] = per_cond_loss
            loss_dict_spectral[settings_name] = per_cond_loss_spectral
            loss_dict_save[settings_name] = float(per_cond_loss.item())
            loss_dict_spectral_save[settings_name] = float(per_cond_loss_spectral.item())
            c += 1
        if self.final_val:
            # losses for metrics dict
            self.calc_vlosses(outputs, targets, set_type, log=False)
            # save audio from best val
            self.save_audio(set_type, total_steps, split_size, inputs, targets, outputs, settings_names)
            # wandb.log({f'{self.lf}_{set_type}_per_cond_best': loss_dict})
            # wandb.log({f'STFT_{set_type}_per_cond_best': loss_dict_spectral})
            # best val losses for metrics dict
            self.metrics_dict[f'{self.lf}_{set_type}_per_cond_best'] = loss_dict_save
            self.metrics_dict[f'STFT_{set_type}_per_cond_best'] = loss_dict_spectral_save
        # else:
        #     # loss per cond (epoch)
        #     # wandb.log({f'{self.lf}_{set_type}_per_cond': loss_dict})
        #     # wandb.log({f'STFT_{set_type}_per_cond': loss_dict_spectral})
        if stage == 'val':
            self.val_inputs, self.val_outputs, self.val_targets = [], [], []
        else:
            # losses for metrics dict
            self.calc_vlosses(outputs, targets, set_type, log=False)
            # save audio from test set
            self.save_audio(set_type, total_steps, split_size, inputs, targets, outputs, settings_names)
            self.test_inputs, self.test_targets, self.test_outputs = [], [], []
            self.metrics_dict[f'{self.lf}_{set_type}_per_cond'] = loss_dict_save
            self.metrics_dict[f'STFT_{set_type}_per_cond'] = loss_dict_spectral_save

    def save_audio(self, set_type, total_steps, split_size, inputs, targets, outputs, settings_names):
        if set_type == 'val':
            selected_segments = self.val_selected
        else:
            selected_segments = self.test_selected
        c = 0
        audio_list = []
        pad = np.zeros(self.truncated_bptt_steps)
        for t in range(0, total_steps, split_size):
            cond_input = inputs[t: t + split_size, :, :]
            cond_target = targets[t: t + split_size, :, :]
            cond_output = outputs[t: t + split_size, :, :]
            settings_string = settings_names[c]
            audio_path = f'./logs/audio/{self.log_subfolder}/{self.model_name}/{set_type}/{settings_string}/'
            if not os.path.exists(audio_path):
                os.makedirs(audio_path)
            for seg in selected_segments:
                seg_list = []
                input_seg = cond_input[seg, :, 0].detach().cpu().numpy()
                output_seg = cond_output[seg, :, 0].detach().cpu().numpy()
                output_seg[0:self.truncated_bptt_steps] = output_seg[0:self.truncated_bptt_steps] * pad
                target_seg = cond_target[seg, :, 0].detach().cpu().numpy()
                target_seg[0:self.truncated_bptt_steps] = target_seg[0:self.truncated_bptt_steps] * pad
                # input segment logging
                input_name = os.path.join(audio_path, f'input_{seg}_{set_type}_{settings_string}.wav')
                seg_list.append(wandb.Audio(input_seg, sample_rate=self.sr, caption=f'seg_{seg}_{settings_string}'))
                sf.write(input_name, input_seg, self.sr)
                # output segment logging
                output_name = os.path.join(audio_path, f'output_{seg}_{set_type}_{settings_string}.wav')
                seg_list.append(wandb.Audio(output_seg, sample_rate=self.sr, caption=f'seg_{seg}_{settings_string}'))
                sf.write(output_name, output_seg, self.sr)
                # target segment logging
                target_name = os.path.join(audio_path, f'target_{seg}_{set_type}_{settings_string}.wav')
                seg_list.append(wandb.Audio(target_seg, sample_rate=self.sr, caption=f'seg_{seg}_{settings_string}'))
                sf.write(target_name, target_seg, self.sr)
                # input, output, target
                audio_list.append(seg_list)
            c += 1
        # audio_table = wandb.Table(columns=['input', 'output', 'target'], data=audio_list)
        # wandb.log({f'audio_{set_type}': audio_table})

    def plot_mag_responses(self):
        settings_names = self.trainer.test_dataloaders[0].dataset.settings_names
        fs = self.sr
        samples = self.truncated_bptt_steps
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
        output_irs = []
        target_irs = []
        for i in range(self.test_inputs[0].shape[0]):
            clean_sweep = self.test_inputs[0][i, :, :]
            output_sweep = self.test_outputs[0][i, :, :]
            target_sweep = self.test_targets[0][i, :, :]
            ir_output = get_ir(clean_sweep.squeeze().cpu().numpy(), output_sweep.squeeze().cpu().numpy())
            ir_target = get_ir(clean_sweep.squeeze().cpu().numpy(), target_sweep.squeeze().cpu().numpy())
            output_irs.append(ir_output)
            target_irs.append(ir_target)
            freq_out, resp_db_out = get_freq_resp(x=ir_output[int(2.5 * fs) - samples: int(2.5 * fs) + samples], fs=fs)
            freq_target, resp_db_target = get_freq_resp(x=ir_target[int(2.5 * fs) - samples: int(2.5 * fs) + samples],
                                                        fs=fs)
            ax1.semilogx(freq_target, resp_db_target, label=f'{settings_names[i]}', linewidth=1)
            ax2.semilogx(freq_out, resp_db_out, label=f'{settings_names[i]}', linewidth=1)
        ax1.set_xlim([20, 20000])
        ax2.set_xlim([20, 20000])
        ax1.set_ylim([-60, -10])
        ax2.set_ylim([-60, -10])
        ax1.grid(which='both')
        ax2.grid(which='both')
        ax1.set_title('Target')
        ax2.set_title('Output')
        ax1.legend(loc='lower right', fontsize=8.5)
        ax2.legend(loc='lower right', fontsize=8.5)
        ax1.set_xlabel('Frequency [Hz]')
        ax2.set_xlabel('Frequency [Hz]')
        ax1.set_ylabel('Magnitude [dB]')
        ax2.set_ylabel('Magnitude [dB]')
        plt.tight_layout()
        wandb.log({'magnitude_responses': wandb.Image(f)})
        plt.close()
        # self.logger.sub_logger.add_figure('magnitude_responses', f, global_step=self.current_epoch, close=True)
        irs_to_save = {'settings_names': settings_names, 'output_irs': output_irs, 'target_irs': target_irs}
        with open(f'./logs/tensorboard/{self.log_subfolder}/{self.model_name}/irs.pkl', 'wb') as file:
            pickle.dump(irs_to_save, file)

    def plot_pot_tapers(self):
        settings = np.linspace(0, 1., 100)
        x_bass = torch.linspace(0, 1., 100, device=self.device).unsqueeze(-1)
        x_mid = torch.linspace(0, 1., 100, device=self.device).unsqueeze(-1)
        x_treble = torch.linspace(0, 1., 100, device=self.device).unsqueeze(-1)
        with torch.no_grad():
            y_bass = self.tone_stack.cond_block.bass_nn(x_bass)
            y_mid = self.tone_stack.cond_block.mid_nn(x_mid)
            y_treble = self.tone_stack.cond_block.treble_nn(x_treble)
        f = plt.figure(figsize=(8, 5))
        plt.title(f'Input: {self.cond_input}, Process: {self.cond_process}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([0., 1.])
        plt.ylim([0., 1.])
        plt.grid()
        plt.plot(settings, y_bass.detach().cpu().numpy().squeeze(), label='BASS')
        plt.plot(settings, y_mid.detach().cpu().numpy().squeeze(), label='MID')
        plt.plot(settings, y_treble.detach().cpu().numpy().squeeze(), label='TREBLE')
        plt.legend()
        plt.tight_layout()
        # self.logger.sub_logger.add_figure('pot_tapers', f, global_step=self.current_epoch, close=True)
        wandb.log({'pot_tapers': wandb.Image(f)})
        plt.close()
        tapers_to_save = {'x': settings,
                          'b': y_bass.detach().cpu().numpy().squeeze(),
                          'm': y_mid.detach().cpu().numpy().squeeze(),
                          't': y_treble.detach().cpu().numpy().squeeze()}
        with open(f'./logs/tensorboard/{self.log_subfolder}/{self.model_name}/tapers.pkl', 'wb') as file:
            pickle.dump(tapers_to_save, file)
