import hashlib
from os import path
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from collections import OrderedDict

from lm_model import GPT2LM
from experiment import experiment


def get_model_name_hash(args):
    imp_args = ["train_size", "train_percent_check", "notation",
                "n_layer", "n_head", "n_positions", "n_embd",
                "seed", "batch_size", "gpus"]
    arg_vals = []
    for arg_name in imp_args:
        arg_vals.append(getattr(args, arg_name))

    str_repr = str(arg_vals).encode("utf-8")
    return f"lm_{hashlib.md5(str_repr).hexdigest()}"


def get_model_name(args):
    arg_to_short_names = OrderedDict(
        [("notation", "not"), ("game_type", "gt"),
         ("train_size", "ts"), ("train_percent_check", "tpr"),
         ("n_layer", "layer"), ("n_head", "head"),
         # ("n_positions", "pos"),  # ("n_embd", "emb"),
         ("stride_size", "stride"), ("window_size", "window"),
         ("real_batch_size", "rbs"),  # ("init_lr", "lr"),
        ]
    )

    str_repr = ""
    for arg_name, short_name in arg_to_short_names.items():
        val = getattr(args, arg_name)
        if val is not None:
            str_repr += short_name + "_" + str(val) + "_"

    str_repr = str_repr.strip('_')

    if args.max_epochs != 10:
        str_repr += f'_epochs_{args.max_epochs}'

    if args.rap_prob:
        str_repr += f"_rp_{int(100 * args.rap_prob)}"
        if args.rap_no_grad:
            str_repr += '_no_grad'

    if args.multiview_margin:
        str_repr += f'_mv_{args.multiview_margin}_{args.multiview_loss_wt}_{args.neg_samples}'

    if args.oracle:
        str_repr += f'_oracle'

    str_repr += f'_seed_{args.seed}'

    return f"lm_{str_repr.strip('_')}"


def main(args):
    seed_everything(args.seed)

    # Set up data_dir, vocab_dir, and model_dir
    base_vocab_dir = path.join(args.base_model_dir, "vocab")
    args.vocab_dir = path.join(base_vocab_dir, "rap" if (args.rap_prob or args.oracle) else "uci")

    args.base_data_dir = path.join(args.base_data_dir, args.game_type)
    args.data_dir = path.join(args.base_data_dir, "uci")

    if args.max_epochs == 1000:
        # Changing the default value
        args.max_epochs = 10

    args.save_dir = args.weights_save_path if args.weights_save_path is not None else args.base_model_dir
    args.model_name = get_model_name(args)

    print(f"Model name: {args.model_name}")
    experiment(args)
    # experiment(args)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Add model args
    parser = GPT2LM.add_model_specific_args(parser)
    # Training args
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--real_batch_size', type=int, default=64)
    parser.add_argument('--init_lr', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--base_data_dir', type=str, default="../data/lm_chess/")
    parser.add_argument('--game_type', type=str, choices=["human", "synthetic"], default="human")
    parser.add_argument('--train_size', type=int, default=100000)
    parser.add_argument('--base_model_dir', type=str, default="../models/")
    parser.add_argument('--notation', type=str, default="uci", choices=['uci'])
    parser.add_argument('--no_other_eval', dest='other_eval', action='store_false')
    parser.add_argument('--seed', type=int, default=42)

    parser = Trainer.add_argparse_args(parser)
    main(parser.parse_args())
