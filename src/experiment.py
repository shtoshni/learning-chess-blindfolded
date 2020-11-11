import sys
import math
from os import path

import torch
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks import EarlyStopping  # , ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from lm_model import GPT2LM
from callbacks.model_checkpoint import MyModelCheckpoint


def experiment(args):
    # Tensorboard logger
    logger = TensorBoardLogger(
        save_dir=args.save_dir,
        version=args.model_name,
        name=None,
    )
    # Callbacks
    lr_logger = LearningRateLogger()

    checkpoint_callback = MyModelCheckpoint(
        verbose=True,
        save_top_k=2,
        period=-1,
        save_last=True,
        prefix='lm_')
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=1,
        verbose=True,
        mode='min'
    )

    # Resume from checkpoint automatically
    resume_from_checkpoint = None
    stop_training = False
    potential_old_checkpoint = path.join(logger.log_dir, 'checkpoints/lm_last.ckpt')
    if path.isfile(potential_old_checkpoint):
        resume_from_checkpoint = potential_old_checkpoint
        print("Resuming training from: ", potential_old_checkpoint)

        last_checkpoint = torch.load(potential_old_checkpoint)
        early_stop_dict = last_checkpoint['early_stop_callback_state_dict']
        print(early_stop_dict)
        if early_stop_dict['wait_count'] > 0:
            print("Early stopping criteria already met, no more training")
            stop_training = True

    sys.stdout.flush()
    args.accumulate_grad_batches = max(args.real_batch_size // args.batch_size, 1)

    trainer = Trainer.from_argparse_args(
        args,
        amp_level='O1',
        gpus=1,
        precision=16,
        # auto_scale_batch_size='binsearch',
        weights_save_path=args.save_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        # val_check_interval=1.0,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=early_stop_callback,
        logger=logger,
        callbacks=[lr_logger],
        reload_dataloaders_every_epoch=True,
        gradient_clip_val=1.0, terminate_on_nan=True,
        row_log_interval=100, log_save_interval=100)

    # Create datamodule
    # data_id_path = OtherCallbacks.get_data_id_path(trainer)
    train_percent_check = 1 if args.train_percent_check is None else args.train_percent_check
    one_epoch_games = int(args.train_size * train_percent_check)
    one_epoch_batches = int(math.ceil(
        one_epoch_games / (args.batch_size * args.accumulate_grad_batches)))
    print(f"One epoch batches: {one_epoch_batches}")

    args.num_training_steps = one_epoch_batches * args.max_epochs
    print("Number of training steps: %d" % args.num_training_steps)

    if not stop_training:
        lm_model = GPT2LM(args, **vars(args))
        trainer.fit(lm_model)
        print(potential_old_checkpoint)
        last_checkpoint = torch.load(potential_old_checkpoint)

    print("Best validation model path: ", last_checkpoint['checkpoint_callback_best_model_path'])
    print("Best validation performance:", last_checkpoint['checkpoint_callback_best_model_score'])

    lm_model = GPT2LM.load_from_checkpoint(checkpoint_path=last_checkpoint['checkpoint_callback_best_model_path'])
    trainer = Trainer.from_argparse_args(
        args,
        amp_level='O1',
        gpus=1,
        precision=16,
        # auto_scale_batch_size='binsearch',
        weights_save_path=args.save_dir,
        # val_check_interval=1.0,
        logger=logger,
        callbacks=[lr_logger],
        reload_dataloaders_every_epoch=True,
        gradient_clip_val=1.0, terminate_on_nan=True,
        row_log_interval=100, log_save_interval=100)

    trainer.test(lm_model)