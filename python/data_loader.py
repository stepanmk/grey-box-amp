import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
import os


class AudioDataset(Dataset):
    def __init__(self,
                 data_dir,
                 file_names,
                 start_sec,
                 end_sec,
                 cond_type,
                 set_type,
                 segment_length_seconds,
                 use_gain_value):
        self.inputs = []
        self.targets = []
        self.settings = []
        self.settings_names = []
        self.fs = 44100
        self.segment_length_samples = int(segment_length_seconds * self.fs)
        self.cond_type = cond_type
        self.set_type = set_type
        self.use_gain_value = use_gain_value

        for i, target_file in enumerate(file_names):
            input_file = target_file[:-14] + 'input.wav'
            # noinspection PyUnresolvedReferences
            inp_data, self.fs = torchaudio.load(os.path.join(data_dir, input_file), channels_first=False)
            # noinspection PyUnresolvedReferences
            tgt_data, self.fs_t = torchaudio.load(os.path.join(data_dir, target_file), channels_first=False)
            settings = None
            if end_sec is None:
                inp_data = inp_data[int(start_sec * self.fs):, :]
                tgt_data = tgt_data[int(start_sec * self.fs):, :]
            else:
                inp_data = inp_data[int(start_sec * self.fs): int(end_sec * self.fs), :]
                tgt_data = tgt_data[int(start_sec * self.fs): int(end_sec * self.fs), :]
            assert (self.fs == self.fs_t)
            self.inputs.append(inp_data)
            self.targets.append(tgt_data)
            settings_string = target_file.split('/')[0].split('_')
            self.settings_names.append('_'.join(settings_string))
            if self.cond_type == 'labels':
                if len(settings_string) == 1:
                    settings = torch.zeros(1)
                    settings[0] = float(settings_string[0][1:]) / 10
                else:
                    if self.use_gain_value:
                        settings = torch.zeros(4)
                        settings[0] = float(settings_string[0][1:]) / 10
                        settings[1] = float(settings_string[1][1:]) / 10
                        settings[2] = float(settings_string[2][1:]) / 10
                        settings[3] = float(settings_string[3][1:]) / 10
                    else:
                        settings = torch.zeros(3)
                        settings[0] = float(settings_string[0][1:]) / 10
                        settings[1] = float(settings_string[1][1:]) / 10
                        settings[2] = float(settings_string[2][1:]) / 10
            if self.cond_type == 'onehot':
                settings = torch.zeros(len(file_names))
                settings[i] = 1.
            self.settings.append(settings)

        self.inputs = torch.stack(self.inputs)
        self.targets = torch.stack(self.targets)
        self.num_segments_in_cond = self.inputs.shape[1] // self.segment_length_samples
        self.num_segments = len(self.settings) * self.num_segments_in_cond
        self.num_conds = len(self.settings)
        self.segs_per_cond = self.num_segments // self.num_conds

    def __getitem__(self, index):
        cond_val = index // self.segs_per_cond
        index = index % self.segs_per_cond
        start = index * self.segment_length_samples
        stop = (index + 1) * self.segment_length_samples
        return self.inputs[cond_val, start:stop, :], self.targets[cond_val, start:stop, :], self.settings[cond_val]

    def __len__(self):
        return self.num_segments


class AudioDataModule(pl.LightningDataModule):
    def __init__(
            self,
            device_name: str = 'JVM',
            data_dir: str = './datasets/final/',
            cond_type: str = 'labels',
            segment_length_seconds: float = 0.5,
            segment_length_seconds_v: float = 5,
            batch_size: int = None,
            batch_size_v: int = None,
            file_names: tuple = (
                # bass at 0, 2, 4, 6, 8, 10
                'B0_M5_T5_G5/B0_M5_T5_G5-speakerout.wav',  # 0
                'B2_M5_T5_G5/B2_M5_T5_G5-speakerout.wav',  # 1
                'B4_M5_T5_G5/B4_M5_T5_G5-speakerout.wav',  # 2
                'B6_M5_T5_G5/B6_M5_T5_G5-speakerout.wav',  # 3
                'B8_M5_T5_G5/B8_M5_T5_G5-speakerout.wav',  # 4
                'B10_M5_T5_G5/B10_M5_T5_G5-speakerout.wav',  # 5

                # mid at 0, 2, 4, 6, 8, 10
                'B5_M0_T5_G5/B5_M0_T5_G5-speakerout.wav',  # 6
                'B5_M2_T5_G5/B5_M2_T5_G5-speakerout.wav',  # 7
                'B5_M4_T5_G5/B5_M4_T5_G5-speakerout.wav',  # 8
                'B5_M6_T5_G5/B5_M6_T5_G5-speakerout.wav',  # 9
                'B5_M8_T5_G5/B5_M8_T5_G5-speakerout.wav',  # 10
                'B5_M10_T5_G5/B5_M10_T5_G5-speakerout.wav',  # 11

                # treble at 0, 2, 4, 6, 8, 10
                'B5_M5_T0_G5/B5_M5_T0_G5-speakerout.wav',  # 12
                'B5_M5_T2_G5/B5_M5_T2_G5-speakerout.wav',  # 13
                'B5_M5_T4_G5/B5_M5_T4_G5-speakerout.wav',  # 14
                'B5_M5_T6_G5/B5_M5_T6_G5-speakerout.wav',  # 15
                'B5_M5_T8_G5/B5_M5_T8_G5-speakerout.wav',  # 16
                'B5_M5_T10_G5/B5_M5_T10_G5-speakerout.wav',  # 17

                # bmt at 0, bmt at 5, bmt at 10
                'B0_M0_T0_G5/B0_M0_T0_G5-speakerout.wav',  # 18
                'B5_M5_T5_G5/B5_M5_T5_G5-speakerout.wav',  # 19
                'B10_M10_T10_G5/B10_M10_T10_G5-speakerout.wav',  # 20
                ),

            file_names_unseen: tuple = (
                # test data
                'B0_M0_T10_G5/B0_M0_T10_G5-speakerout.wav',
                'B0_M10_T0_G5/B0_M10_T0_G5-speakerout.wav',
                'B0_M10_T10_G5/B0_M10_T10_G5-speakerout.wav',
                'B10_M0_T0_G5/B10_M0_T0_G5-speakerout.wav',
                'B10_M0_T10_G5/B10_M0_T10_G5-speakerout.wav',
                'B10_M10_T0_G5/B10_M10_T0_G5-speakerout.wav',
                'B6.5_M8.5_T3.5_G5/B6.5_M8.5_T3.5_G5-speakerout.wav',
                'B1_M3_T7_G5/B1_M3_T7_G5-speakerout.wav',
                'B3_M7_T1_G5/B3_M7_T1_G5-speakerout.wav',
            ),
            leave_from_train: tuple = tuple(range(0, 18)),
            reduce_train_set: bool = False,
            train_secs: tuple = (0, 240),
            val_secs: tuple = (240, 300),
            test_secs: tuple = (300, 360),
            file_idx: int = None,
            file_idx_test: int = None,
            use_gain_value: bool = False,
    ):
        super().__init__()
        self.cond_type = cond_type
        self.data_dir = data_dir
        self.segment_length_samples = segment_length_seconds
        self.batch_size = batch_size
        self.segment_length_samples_v = segment_length_seconds_v
        self.batch_size_v = batch_size_v
        self.device_name = device_name
        self.datasets = {}
        self.file_names = file_names
        self.file_names_unseen = file_names_unseen
        self.train_secs = train_secs
        self.val_secs = val_secs
        self.test_secs = test_secs
        self.file_idx = file_idx
        self.file_idx_test = file_idx_test
        self.leave_from_train = list(leave_from_train) if leave_from_train is not None else None
        self.reduce_train_set = reduce_train_set
        self.use_gain_value = use_gain_value

    def setup(self, stage):
        if self.file_idx is not None:
            self.file_names = [self.file_names[self.file_idx]]

        if self.file_idx_test is not None:
            self.file_names_unseen = [self.file_names_unseen[self.file_idx_test]]

        file_names = list(self.file_names).copy()
        if self.reduce_train_set and self.leave_from_train is not None:
            for val in self.leave_from_train:
                file_names.remove(self.file_names[val])

        self.datasets['train'] = AudioDataset(data_dir=self.data_dir,
                                              file_names=file_names,
                                              start_sec=self.train_secs[0],
                                              end_sec=self.train_secs[1],
                                              segment_length_seconds=self.segment_length_samples,
                                              cond_type=self.cond_type,
                                              set_type='train',
                                              use_gain_value=self.use_gain_value)
        self.datasets['val'] = AudioDataset(data_dir=self.data_dir,
                                            file_names=file_names,
                                            start_sec=self.val_secs[0],
                                            end_sec=self.val_secs[1],
                                            segment_length_seconds=self.segment_length_samples_v,
                                            cond_type=self.cond_type,
                                            set_type='val',
                                            use_gain_value=self.use_gain_value)
        self.datasets['test'] = AudioDataset(data_dir=self.data_dir,
                                             file_names=file_names,
                                             start_sec=self.test_secs[0],
                                             end_sec=self.test_secs[1],
                                             segment_length_seconds=self.segment_length_samples_v,
                                             cond_type=self.cond_type,
                                             set_type='test',
                                             use_gain_value=self.use_gain_value)

        if self.file_names_unseen is not None:
            self.datasets['test_unseen'] = AudioDataset(data_dir='./datasets/final/test',
                                                        file_names=self.file_names_unseen,
                                                        start_sec=self.test_secs[0],
                                                        end_sec=self.test_secs[1],
                                                        segment_length_seconds=self.segment_length_samples_v,
                                                        cond_type=self.cond_type,
                                                        set_type='test_unseen',
                                                        use_gain_value=self.use_gain_value)
            self.datasets['test_sweeps'] = AudioDataset(data_dir='./datasets/final/test',
                                                        file_names=self.file_names_unseen,
                                                        start_sec=375,
                                                        end_sec=380,
                                                        segment_length_seconds=self.segment_length_samples_v,
                                                        cond_type=self.cond_type,
                                                        set_type='test_sweeps',
                                                        use_gain_value=self.use_gain_value)

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size,
                          shuffle=True, num_workers=0, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size_v,
                          shuffle=False, num_workers=0, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size_v,
                          shuffle=False, num_workers=0, drop_last=True)

    def test_unseen_dataloader(self):
        return DataLoader(self.datasets['test_unseen'], batch_size=self.batch_size_v,
                          shuffle=False, num_workers=0, drop_last=True)

    def test_sweeps_dataloader(self):
        return DataLoader(self.datasets['test_sweeps'], batch_size=len(self.file_names_unseen),
                          shuffle=False, num_workers=0, drop_last=True)
