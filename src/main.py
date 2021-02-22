import hashlib
from os import path
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from collections import OrderedDict

from lm_model import ChessLM
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
    if args.model_type in ['reformer', 'performer', 'transformer']:
        arg_to_short_names = OrderedDict(
            [("train_size", "ts"), ("train_percent_check", "tpr"),
             ("n_layer", "layer"), ("n_head", "head"), ("n_embd", "emb"),
             ("stride_size", "stride"), ("window_size", "window"),
             ("real_batch_size", "rbs"),
             ]
        )

        str_repr = ""
        if args.model_type != 'transformer':
            str_repr = args.model_type  + "_"
        for arg_name, short_name in arg_to_short_names.items():
            val = getattr(args, arg_name)
            if val is not None:
                str_repr += short_name + "_" + str(val) + "_"

        str_repr = str_repr.strip('_')
        if args.max_epochs != 10:
            str_repr += f'_epochs_{args.max_epochs}'

        if args.model_type == 'transformer':
            # We train the different model variants only for the standard transformer
            if args.rap_prob:
                str_repr += f"_rp_{int(100 * args.rap_prob)}"
                if args.rap_no_grad:
                    str_repr += '_no_grad'

            if args.oracle:
                str_repr += '_oracle'

        elif args.model_type == 'reformer':
            if args.local_window_size != 50:
                str_repr += f"_lws_{args.local_window_size}"
            if args.num_buckets is not None:
                str_repr += f"_nb_{args.num_buckets}"
            if args.num_hashes != 1:
                str_repr += f"_nh_{args.num_hashes}"

        elif args.model_type == 'performer':
            if args.local_window_size != 50:
                str_repr += f"_lws_{args.local_window_size}"
            if args.generalized_attention:
                str_repr += f"_general"
            if args.feature_redraw != 1000:
                str_repr += f"_redraw_{args.feature_redraw}"
            if args.local_attn_heads != 6:
                str_repr += f"_local_{args.local_attn_heads}"

    elif args.model_type == 'rnn':
        arg_to_short_names = OrderedDict(
            [("train_size", "ts"), ("train_percent_check", "tpr"),
             ("n_layer", "layer"), ("n_hid", "hid"),  ("n_embd", "emb"),
             ("rnn_type", "rnn"), ("rnn_dropout", "drop"),
             ("real_batch_size", "rbs"),
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

    else:
        raise NotImplementedError(f'Model type {args.model_type} not supported')

    str_repr += f'_seed_{args.seed}'

    return f"lm_{str_repr.strip('_')}"


def main(args):
    seed_everything(args.seed)

    # Set up data_dir, vocab_dir, and model_dir
    args.vocab_dir = path.join(args.data_dir, "vocab/uci")
    # Change data directory to point to UCI directory
    args.data_dir = path.join(args.data_dir, "uci")

    if args.max_epochs == 1000:
        # Changing the default value
        args.max_epochs = 10

    args.save_dir = args.weights_save_path if args.weights_save_path is not None else args.base_model_dir
    args.model_name = get_model_name(args)

    print(f"Model name: {args.model_name}")
    experiment(args)


if __name__ == '__main__':
    parser = ArgumentParser()

    # Add model args
    parser = ChessLM.add_model_specific_args(parser)
    # Training args
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--real_batch_size', type=int, default=64)
    parser.add_argument('--init_lr', type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--data_dir', type=str, default="../data/lm_chess/")
    parser.add_argument('--train_size', type=int, default=200000)
    parser.add_argument('--base_model_dir', type=str, default="../models/")
    parser.add_argument('--no_other_eval', dest='other_eval', default=True, action='store_false')
    parser.add_argument('--seed', type=int, default=42)

    parser = Trainer.add_argparse_args(parser)
    main(parser.parse_args())
