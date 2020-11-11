import torch
import os

from os import path
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import _logger as log
from transformers import GPT2Config, get_linear_schedule_with_warmup
from model_utils.gpt2_model import GPT2LMHeadModel
from model_utils.utils import get_strided_attn_mask, get_last_window_attn_mask
from chess_utils.parsing_utils import return_valid_prefix
from constants import GAME_TYPES
from data_utils.simple_datamodule import ChessLMDataModule


os.environ["TOKENIZERS_PARALLELISM"] = "true"


class GPT2LM(LightningModule):
    def __init__(self, args,
                 n_embd=768, n_positions=1024, n_layer=12, n_head=12,
                 init_lr=3e-4, notation="lan", accumulate_grad_batches=1,
                 num_training_steps=None, other_eval=False,
                 stride_size=None, window_size=None,
                 grounding_model='conv', multiview_margin=None, multiview_loss_wt=1.0, neg_samples=1,
                 oracle=False, state_dropout=0.1,
                 num_conv_filters=64, num_fc_layers=1, kernel_size=2, out_channels=300,
                 model_type='transformer',
                 **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.model_type = model_type  # RNN vs Transformer

        # Board state setting
        self.multiview = (multiview_margin is not None)
        self.multiview_loss_wt = multiview_loss_wt
        self.oracle = oracle
        self.grounding = self.oracle or self.multiview

        self.datamodule = ChessLMDataModule(multiview=self.multiview, grounding=self.grounding, **vars(args))
        self.tokenizer = self.datamodule.tokenizer
        vocab_size = len(self.tokenizer.get_vocab())

        if self.model_type == 'transformer':
            self.config = GPT2Config(
                vocab_size=vocab_size,
                n_embd=n_embd,
                n_positions=n_positions,
                n_ctx=n_positions,
                n_head=n_head,
                n_layer=n_layer,
            )
            if stride_size is not None:
                attention_mask = get_strided_attn_mask(stride_size, max_seq_length=n_positions)
                self.config.static_attention_mask = attention_mask
            elif window_size is not None:
                attention_mask = get_last_window_attn_mask(window_size, max_seq_length=n_positions)
                self.config.static_attention_mask = attention_mask
            else:
                self.config.static_attention_mask = None

            # Whether to use board state or not
            self.config.grounding_model = grounding_model
            # Only training
            self.config.multiview = self.multiview
            self.config.multiview_margin = multiview_margin
            self.config.neg_samples = neg_samples
            self.config.state_dropout = state_dropout

            # Both inference and training
            self.config.oracle = oracle

            # Convolution stuff
            self.config.num_conv_filters = num_conv_filters
            self.config.num_fc_layers = num_fc_layers
            self.config.kernel_size = kernel_size
            self.config.out_channels = out_channels

            self.max_len = n_positions
            self.notation = notation
            self.model = GPT2LMHeadModel(config=self.config)

            self.init_lr = init_lr
            self.accumulate_grad_batches = accumulate_grad_batches
            self.num_training_steps = num_training_steps
            self.other_eval = other_eval
            self.other_eval_log_dir = None

            self.current_epoch_steps = 0

        else:
            pass

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('-model_type', type=str, default='transformer', choices= ['transformer', 'rnn'])
        # RNN arguments - TODO implement RNN language model
        parser.add_argument('-rnn_type', type=str, default='lstm', choices=['gru', 'lstm'])
        parser.add_argument('-rnn_dropout', type=float, default=0.5)
        # Transformer arguments
        parser.add_argument('--n_layer', type=int, default=12)
        parser.add_argument('--n_head', type=int, default=12)
        parser.add_argument('--n_positions', type=int, default=512)
        parser.add_argument('--n_embd', type=int, default=768)
        parser.add_argument('--stride_size', type=int, default=None)
        parser.add_argument('--window_size', type=int, default=None)
        # Adding board state
        # (0) RAP
        parser.add_argument('--rap_prob', type=float, default=0.0)
        parser.add_argument('--rap_no_grad', default=False, action="store_true")
        # (1) Multiview
        parser.add_argument('--multiview_margin', type=float, default=None)
        parser.add_argument('--multiview_loss_wt', type=float, default=1.0)
        # (2) Oracle
        parser.add_argument('--oracle', default=False, action="store_true")
        # Boars state model details
        parser.add_argument('--grounding_model', type=str, default='conv', choices=['other_mlp', 'my_mlp', 'conv'])
        parser.add_argument('--kernel_size', type=int, default=2)
        parser.add_argument('--num_conv_filters', type=int, default=128)
        parser.add_argument('--out_channels', type=int, default=128)
        parser.add_argument('--num_fc_layers', type=int, default=1)
        parser.add_argument('--state_dropout', type=float, default=0.1)
        parser.add_argument('--neg_samples', type=int, default=10)
        # Fixed attention head - TODO implement fixed attention head
        parser.add_argument('--fixed_attention', default=False, action="store_true")

        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.init_lr)
        linear_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0.1 * self.num_training_steps,
            num_training_steps=self.num_training_steps)
        scheduler = {'scheduler': linear_scheduler, 'interval': 'step'}
        return [optimizer], [scheduler]

    def forward(self, batch):
        outputs = self.model(**batch)
        return outputs[:2]

    def training_step(self, batch, batch_ids):
        # print(batch_ids)
        if not self.multiview:
            loss, _ = self(batch)
            max_margin_loss = None
        else:
            loss, max_margin_loss = self(batch)

        train_loss = loss
        train_log = {'loss/train_loss': loss}
        if max_margin_loss is not None:
            train_loss = loss + self.multiview_loss_wt * max_margin_loss
            train_log['loss/train_max_margin'] = max_margin_loss

        self.current_epoch_steps += batch["input_ids"].shape[0]
        return {'loss': train_loss, 'log': train_log}

    def validation_step(self, batch, batch_ids, split="val"):
        # print("Validation", batch_ids)
        input_ids, labels = batch["input_ids"], batch["labels"]
        loss, _ = self(batch)
        batch_tokens = torch.sum(input_ids != self.tokenizer.pad_token_id) - input_ids.shape[0]
        val_log = {f'loss/{split}_loss': loss.detach()}
        return {f'{split}_loss': loss.detach(), 'log': val_log, 'batch_tokens': batch_tokens,
                'batch_size': input_ids.shape[0]}

    def test_step(self, batch, batch_ids):
        return self.validation_step(batch, batch_ids, split='test')

    def generate(self, prefix=None, num_beams=1, max_len=None):
        tokenizer = self.tokenizer

        prefix = [tokenizer.bos_token_id] if prefix is None else prefix
        max_len = self.max_len if max_len is None else max_len

        output = self.model.generate(
            input_ids=torch.tensor([prefix]).to(self.device),
            bos_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_beams=num_beams, temperature=1.0, repetition_penalty=1.0, max_length=max_len)

        return output.tolist()[0]

    def generate_sample_game(self):
        """Log a sample game."""
        tokenizer = self.tokenizer

        sample_game = self.generate()
        sample_game = tokenizer.decode(sample_game, keep_special_tokens=False)
        valid_prefix, move_counter = return_valid_prefix(sample_game, notation=self.notation)

        self.logger.experiment.add_scalar("Other Eval/Game Length", move_counter, self.current_epoch)
        self.logger.experiment.add_text('valid_prefix', ' '.join(valid_prefix), self.current_epoch)
        self.logger.experiment.add_text('sample_game', sample_game, self.current_epoch)

    def evaluate_on_other_eval_sets(self, to_log=True):
        self.other_eval_log_dir = path.join(self.logger.log_dir, "other_eval")
        if not path.exists(self.other_eval_log_dir):
            os.makedirs(self.other_eval_log_dir)

        for game_type in GAME_TYPES:
            for eval_type, eval_set in self.datamodule.other_eval_sets[game_type].items():
                eval_type_file_name = f'{game_type}_{"_".join([part.lower() for part in eval_type.split()])}.log'
                log_file = path.join(self.other_eval_log_dir, eval_type_file_name)

                with open(log_file, 'w') as f:
                    pred_corr_dict = OrderedDict()
                    offset = 1
                    _, _, gt_option_dict, _ = eval_set[0]
                    output_cols = ['prefix']
                    for cloze_field in gt_option_dict:
                        output_cols.append(cloze_field)
                        output_cols.append('acc')
                    output_cols.append('pred')
                    f.write(','.join(output_cols) + '\n')

                    for prefix, board_rep_aligned, gt_option_dict, prefix_str in eval_set:
                        if self.grounding:
                            logits = self.model(input_ids=torch.tensor([prefix]).cuda(),
                                                board_rep=board_rep_aligned.cuda())[0]
                        else:
                            logits = self.model(input_ids=torch.tensor([prefix]).cuda())[0]

                        prediction = tuple([torch.argmax(logits[0, -1, :], dim=0).item()])

                        prediction_str = self.tokenizer.decode(prediction)

                        output_vals = [prefix_str]
                        for cloze_field, (gt_options_enc, gt_str) in gt_option_dict.items():
                            instance_corr = 0
                            if cloze_field not in pred_corr_dict:
                                pred_corr_dict[cloze_field] = 0
                            if prediction in gt_options_enc:
                                pred_corr_dict[cloze_field] += 1
                                instance_corr = 1

                            output_vals.append(gt_str)
                            output_vals.append(str(instance_corr))

                        output_vals.append(prediction_str)
                        f.write(','.join(output_vals) + '\n')

                for cloze_field, pred_corr in pred_corr_dict.items():
                    accuracy = pred_corr * 100/len(eval_set)
                    specific_eval_type = f'{eval_type} {cloze_field.capitalize()}'
                    log.info(f'\n{game_type.capitalize()} - {specific_eval_type}: {accuracy}')
                    if to_log:
                        self.logger.experiment.add_scalar(f"{game_type.capitalize()}/" + specific_eval_type,
                                                          accuracy, self.current_epoch)

    def validation_epoch_end(self, outputs, split='val'):
        if self.config.n_layer >= 6:
            # Otherwise debugging
            if (self.global_step > 0 or split == 'test') and self.other_eval:
                self.evaluate_on_other_eval_sets()

        # Calculate validation loss
        val_loss = torch.stack([x[f'{split}_loss'] for x in outputs])
        val_tokens = torch.stack([x['batch_tokens'] for x in outputs])
        num_examples = sum([x['batch_size'] for x in outputs])
        canonical_tokens = self.datamodule.num_of_canonical_tokens[split][:num_examples]

        avg_log_p = torch.sum(val_loss * val_tokens)/torch.sum(val_tokens)
        canonical_log_p = torch.sum(val_loss * val_tokens)/sum(canonical_tokens)

        val_log = {f'loss/{split}_loss': avg_log_p, f'loss/canonical_{split}_loss': canonical_log_p}
        print("\nLog-loss: %.3f, Canonical log loss: %.3f" % (avg_log_p, canonical_log_p))
        return {'log': val_log, f'{split}_loss': avg_log_p, f'canonical_{split}_loss': canonical_log_p}

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, split='test')
