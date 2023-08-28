# import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
# noinspection PyProtectedMember
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.tensorboard import SummaryWriter

from data_loader import AudioDataModule
from models.amp_model import AmpModel

import warnings
import json

warnings.filterwarnings('ignore', '.*does not have many workers.*')
warnings.filterwarnings('ignore', '.*Checkpoint directory.*')

prsr = argparse.ArgumentParser()

prsr.add_argument('--train_lf', '-lf', default='ESR')

prsr.add_argument('--cond_type', '-ct', default='labels')
prsr.add_argument('--cond_process', '-cp', default='pot')

prsr.add_argument('--pre_rnn', '-prn', default='LSTM')
prsr.add_argument('--pre_hidden', '-prh', default=40, type=int)
prsr.add_argument('--pre_res', '-prs', default=1, type=int)

prsr.add_argument('--poweramp', '-pa', default=1, type=int)
prsr.add_argument('--power_rnn', '-pon', default='GRU')
prsr.add_argument('--power_hidden', '-poh', default=8, type=int)

prsr.add_argument('--freeze_cond', '-fc', default=0, type=int)

prsr.add_argument('--max_epochs', '-me', default=300, type=int)

prsr.add_argument('--bs_train', '-bt', default=80, type=int)
prsr.add_argument('--bs_val', '-bv', default=12, type=int)

prsr.add_argument('--rnn_only', '-ro', default=0, type=int)

prsr.add_argument('--reduce_train', '-rt', default=1, type=int)
prsr.add_argument('--reduce_targets', '-rtg', default=1, type=int)
prsr.add_argument('--single_target', '-st', default=0, type=int)
prsr.add_argument('--target_file', '-tf', default=0, type=int)

prsr.add_argument('--log_subfolder', '-ls', default='amp_model')

args = prsr.parse_args()


if __name__ == "__main__":
    n_targets = None
    # 3 targets
    if args.reduce_targets == 0:
        leave_from_train = tuple(range(0, 18))
        n_targets = 3
    # 15 targets
    elif args.reduce_targets == 1:
        leave_from_train = (2, 3, 8, 9, 14, 15)
        n_targets = 15
    elif args.reduce_targets == 3:
        leave_from_train = (1, 2, 3, 4, 7, 8, 9, 10, 13, 14, 15, 16)
        n_targets = 9
    # 21 targets
    else:
        leave_from_train = None
        n_targets = 21

    # dataset setup
    data = AudioDataModule(batch_size=args.bs_train,
                           batch_size_v=args.bs_val,
                           cond_type=args.cond_type,
                           segment_length_seconds=0.5,
                           segment_length_seconds_v=5,
                           reduce_train_set=args.reduce_train,
                           leave_from_train=leave_from_train,
                           file_idx=args.target_file if args.single_target else None)
    data.setup(stage='fit')

    # create name for save directory
    if args.single_target:
        save_dir = '1_target_'
    else:
        save_dir = f'{n_targets}_targets_'
    if args.rnn_only:
        save_dir += f'{args.pre_rnn}{args.pre_hidden}_rnn_only'
    else:
        save_dir += f'{args.pre_rnn}{args.pre_hidden}'
    if args.poweramp:
        save_dir += f'_pwr_{args.power_rnn}{args.power_hidden}'

    # train, val, test
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    test_dataloader = data.test_dataloader()
    test_unseen_dataloader = data.test_unseen_dataloader()
    test_sweeps_dataloader = data.test_sweeps_dataloader()

    num_conds = data.datasets['train'].num_conds

    model = AmpModel(train_lf=args.train_lf,
                     pre_type=args.pre_rnn,
                     pre_hidden=args.pre_hidden,
                     pre_res=args.pre_res,
                     batch_size=args.bs_train,
                     batch_size_val=args.bs_val,
                     model_name=save_dir,
                     cond_input=args.cond_type,
                     cond_process=args.cond_process,
                     n_targets=num_conds,
                     rnn_only=args.rnn_only,
                     log_subfolder=args.log_subfolder,
                     include_poweramp=args.poweramp,
                     freeze_cond_block=args.freeze_cond)

    # name log dir and create logger and early stopping callback
    logs_dir = f'./logs/tensorboard/{args.log_subfolder}'

    wandb_logger = WandbLogger(project='grey-box-amp', name=save_dir)

    early_stopping = EarlyStopping(f'{args.train_lf}_val', patience=15)
    checkpoint_callback = ModelCheckpoint(dirpath=logs_dir + '/' + save_dir, every_n_epochs=1, save_top_k=-1,
                                          monitor=f'{args.train_lf}_val')

    logger = TensorBoardLogger(save_dir=logs_dir, name=save_dir, version=1)
    logger.sub_logger = SummaryWriter(logger.log_dir)

    # create trainer object
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         check_val_every_n_epoch=2,
                         callbacks=[checkpoint_callback, early_stopping],
                         logger=[logger, wandb_logger],
                         accelerator='gpu',
                         devices=1,
                         num_sanity_val_steps=0,
                         log_every_n_steps=5)

    # train model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # save path of the best model checkpoint
    best_path = trainer.checkpoint_callback.best_model_path
    with open(logs_dir + '/' + save_dir + '/version_1/best_model_path.txt', 'w') as f:
        f.write(best_path)

    # run validation again to save sounds from the best model
    model.final_val = True
    trainer.validate(model=model, dataloaders=val_dataloader, ckpt_path=best_path)

    # model testing
    model.final_val = False
    trainer.test(model=model, dataloaders=test_dataloader, ckpt_path=best_path)
    trainer.test(model=model, dataloaders=test_unseen_dataloader, ckpt_path=best_path)

    with open(f'./logs/tensorboard/{args.log_subfolder}/{model.model_name}/metrics.json', 'w') as file:
        json.dump(model.metrics_dict, file)
