import subprocess


def run_config(d):
    cmd = f'C:/Users/Stepan/miniconda3/envs/gray-box-amp/python train_model.py' \
          f' --train_lf {d["train_lf"]}' \
          f' --pre_rnn {d["pre_rnn"]}' \
          f' --pre_hidden {d["pre_hidden"]}' \
          f' --pre_res {d["pre_res"]}' \
          f' --poweramp {d["poweramp"]}' \
          f' --power_rnn {d["power_rnn"]}' \
          f' --power_hidden {d["power_hidden"]}' \
          f' --rnn_only {d["rnn_only"]} ' \
          f' --cond_type {d["cond_type"]} ' \
          f' --cond_process {d["cond_process"]} ' \
          f' --freeze_cond {d["freeze_cond"]}' \
          f' --max_epochs {d["max_epochs"]} ' \
          f' --reduce_train {d["reduce_train"]}' \
          f' --reduce_targets {d["reduce_targets"]}' \
          f' --single_target {d["single_target"]}' \
          f' --target_file {d["target_file"]}'
    subprocess.run(cmd)


# preamp, ts, poweramp, single target (all controls at 5), frozen nonlinear mappings of the pots
cfg1 = {'train_lf': 'ESR',
        'pre_rnn': 'LSTM',
        'pre_hidden': 40,
        'poweramp': 1,
        'power_rnn': 'GRU',
        'power_hidden': 8,
        'pre_res': 1,
        'rnn_only': 0,
        'cond_type': 'labels',
        'cond_process': 'pot',
        'freeze_cond': 1,
        'max_epochs': 350,
        'reduce_train': 0,
        'reduce_targets': 1,
        'single_target': 1,
        'target_file': 19}

# preamp, ts, poweramp, 3 targets (all at 0, all at 5, all at 10)
cfg2 = {'train_lf': 'ESR',
        'pre_rnn': 'LSTM',
        'pre_hidden': 40,
        'pre_res': 1,
        'poweramp': 1,
        'power_rnn': 'GRU',
        'power_hidden': 8,
        'rnn_only': 0,
        'cond_type': 'labels',
        'cond_process': 'pot',
        'freeze_cond': 0,
        'max_epochs': 350,
        'reduce_train': 1,
        'reduce_targets': 0,
        'single_target': 0,
        'target_file': 0}

# preamp, ts, poweramp, 15 targets (BMT varied from 0 to 10 --- 0, 2, 8, 10)
cfg3 = {'train_lf': 'ESR',
        'pre_rnn': 'LSTM',
        'pre_hidden': 40,
        'pre_res': 1,
        'poweramp': 1,
        'power_rnn': 'GRU',
        'power_hidden': 8,
        'rnn_only': 0,
        'cond_type': 'labels',
        'cond_process': 'pot',
        'freeze_cond': 0,
        'max_epochs': 350,
        'reduce_train': 1,
        'reduce_targets': 1,
        'single_target': 0,
        'target_file': 0}

# preamp, ts, poweramp, 21 targets (BMT varied from 0 to 10 --- 0, 2, 4, 6, 8, 10)
cfg4 = {'train_lf': 'ESR',
        'pre_rnn': 'LSTM',
        'pre_hidden': 40,
        'pre_res': 1,
        'poweramp': 1,
        'power_rnn': 'GRU',
        'power_hidden': 8,
        'rnn_only': 0,
        'cond_type': 'labels',
        'cond_process': 'pot',
        'freeze_cond': 0,
        'max_epochs': 350,
        'reduce_train': 1,
        'reduce_targets': 2,
        'single_target': 0,
        'target_file': 0}

# rnn only (LSTM 48), 1 target (all controls at 5)
cfg5 = {'train_lf': 'ESR',
        'pre_rnn': 'LSTM',
        'pre_hidden': 48,
        'pre_res': 1,
        'poweramp': 0,
        'power_rnn': 'GRU',
        'power_hidden': 8,
        'rnn_only': 1,
        'cond_type': 'labels',
        'cond_process': 'pot',
        'freeze_cond': 0,
        'max_epochs': 350,
        'reduce_train': 0,
        'reduce_targets': 2,
        'single_target': 1,
        'target_file': 19}

# rnn only (LSTM 48), 3 targets (all at 0, all at 5, all at 10)
cfg6 = {'train_lf': 'ESR',
        'pre_rnn': 'LSTM',
        'pre_hidden': 48,
        'pre_res': 1,
        'poweramp': 0,
        'power_rnn': 'GRU',
        'power_hidden': 8,
        'rnn_only': 1,
        'cond_type': 'labels',
        'cond_process': 'pot',
        'freeze_cond': 0,
        'max_epochs': 350,
        'reduce_train': 1,
        'reduce_targets': 0,
        'single_target': 0,
        'target_file': 0}

# rnn only (LSTM 48), 15 targets (BMT varied from 0 to 10 --- 0, 2, 8, 10)
cfg7 = {'train_lf': 'ESR',
        'pre_rnn': 'LSTM',
        'pre_hidden': 48,
        'pre_res': 1,
        'poweramp': 0,
        'power_rnn': 'GRU',
        'power_hidden': 8,
        'rnn_only': 1,
        'cond_type': 'labels',
        'cond_process': 'pot',
        'freeze_cond': 0,
        'max_epochs': 350,
        'reduce_train': 1,
        'reduce_targets': 1,
        'single_target': 0,
        'target_file': 0}

# rnn only (LSTM 48), 21 targets (BMT varied from 0 to 10 --- 0, 2, 4, 6, 8, 10)
cfg8 = {'train_lf': 'ESR',
        'pre_rnn': 'LSTM',
        'pre_hidden': 48,
        'pre_res': 1,
        'poweramp': 0,
        'power_rnn': 'GRU',
        'power_hidden': 8,
        'rnn_only': 1,
        'cond_type': 'labels',
        'cond_process': 'pot',
        'freeze_cond': 0,
        'max_epochs': 350,
        'reduce_train': 1,
        'reduce_targets': 2,
        'single_target': 0,
        'target_file': 0}
