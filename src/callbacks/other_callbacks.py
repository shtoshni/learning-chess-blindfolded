from pytorch_lightning.callbacks.base import Callback
import torch
from os import path
import os
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only
from pytorch_lightning import _logger as log
import math

class OtherCallbacks(Callback):
    def __init__(self, freq=100):
        super().__init__()
        self.freq = freq
        self.counter = 0

    def on_batch_end(self, trainer, pl_module):
        if self.freq > 0:
            if pl_module.global_step % self.freq == 0 and (self.counter % pl_module.accumulate_grad_batches == 0):
                print("Max memory: %.3f GB" % (torch.cuda.max_memory_allocated(pl_module.device) / 1024 ** 3))
                torch.cuda.reset_peak_memory_stats()
        self.counter += 1

    @staticmethod
    def get_data_id_path(trainer):
        return path.join(trainer.logger.save_dir, trainer.logger.version, "train_sampler.pth")

    def save_data_ids(self, trainer, pl_module):
        rem_examples = pl_module.datamodule.shuffled_train_order[pl_module.current_epoch_steps:]
        save_path = self.get_data_id_path(trainer)

        if len(rem_examples) > 0 and (pl_module.current_epoch_steps > 0):
            # Check that training hasn't just started i.e. start of new epoch and there are remaining examples
            # print("Remaining examples:", rem_examples)
            log.info(f'Saving training order of {len(rem_examples)} examples')
            torch.save(rem_examples, save_path)
        else:
            if path.exists(save_path):
                os.remove(save_path)

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        self.save_data_ids(trainer, pl_module)

    @rank_zero_only
    def on_epoch_end(self, trainer, pl_module):
        pl_module.current_epoch_steps = 0

        save_path = self.get_data_id_path(trainer)
        if path.exists(save_path):
            os.remove(save_path)

        log.info(f'Removed train ID file at end of epoch')

