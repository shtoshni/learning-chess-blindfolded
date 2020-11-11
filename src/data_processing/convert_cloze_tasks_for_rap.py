import argparse
import random
import numpy as np
import os
from os import path
import glob

from chess_utils.conversion_utils import convert_lan_to_uci
from chess_utils.utils import get_move_prefix_size, get_syntax_moves, get_syntax_locations
from collections import OrderedDict, defaultdict



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-base_dir', type=str, default=None,
                        help="Base data directory where all notation specific directories are present.")
    parser.add_argument('-rap_prob_list', type=str, default="0.01, 0.05, 0.15, 0.85, 1.0",
                        help="RAP probability list specified as comma separate floating numbers.")

    args = parser.parse_args()
    args.rap_prob_list = [float(rap_prob) for rap_prob in args.rap_prob_list.split(",")]
    return args


def convert_lan_uci_to_rap(lan_game, uci_game, rap_prob=0.15):
    random_vec = np.random.choice([0, 1], size=(len(lan_game),), p=[1 - rap_prob, rap_prob])
    rap_moves = []
    for random_val, lan_move, uci_move in zip(random_vec, lan_game, uci_game):
        if random_val and lan_move[0] != 'O':
            rap_move = lan_move[0] + uci_move
        else:
            rap_move = uci_move
        rap_moves.append(rap_move)

    return rap_moves


def parse_line(line, fields):
    instance = line.strip().split(",")
    prefix = instance[0]
    moves = prefix.split()
    game_prefix, move_prefix = moves[:-1], moves[-1]

    field_to_options_dict = OrderedDict()
    for (field, options) in zip(fields[1:], instance[1:]):
        field_to_options_dict[field] = options

    return game_prefix, move_prefix, field_to_options_dict


def get_end_position_eval_data(args):
    lan_eval_files = glob.glob(path.join(args.base_dir, "lan_with_p/dev_end_*"))
    uci_dir = path.join(args.base_dir, "uci")
    output_dir_template = path.join(args.base_dir, "rap_{}")

    for rap_prob in args.rap_prob_list:
        random.seed(42)
        np.random.seed(42)

        output_dir = output_dir_template.format(int(rap_prob * 100))
        print(output_dir)
        if not path.exists(output_dir):
            os.makedirs(output_dir)

        for lan_file in lan_eval_files:
            basename = path.basename(lan_file)
            uci_file = path.join(uci_dir, basename)
            output_file = path.join(output_dir, basename)

            ignore_first = True
            fields = None
            with open(lan_file) as f, open(uci_file) as g, open(output_file, 'w') as writer:
                for lan_line, uci_line in zip(f, g):
                    if ignore_first:
                        ignore_first = False
                        fields = lan_line.strip().split(",")
                        writer.write(lan_line)
                        # print()
                        continue

                    lan_game_prefix, lan_move_prefix, lan_options_dict = parse_line(lan_line, fields)
                    uci_game_prefix, uci_move_prefix, uci_options_dict = parse_line(uci_line, fields)

                    assert (len(lan_options_dict) == len(uci_options_dict))
                    rap_game_prefix = convert_lan_uci_to_rap(lan_game_prefix, uci_game_prefix, rap_prob=rap_prob)

                    move_prefix_prob = random.uniform(0, 1)
                    if move_prefix_prob <= rap_prob:
                        rap_move_prefix = lan_move_prefix
                    else:
                        rap_move_prefix = uci_move_prefix

                    instance_str = ' '.join(rap_game_prefix) + " " + rap_move_prefix
                    for field, move_options in uci_options_dict.items():
                        instance_str += f",{move_options}"
                    # print(rap_move_options_dict)
                    writer.write(instance_str + "\n")


def get_start_position_eval_data(args):
    lan_eval_files = glob.glob(path.join(args.base_dir, "lan_with_p/dev_start_*"))
    output_dir_template = path.join(args.base_dir, "rap_{}")
    for rap_prob in args.rap_prob_list:
        random.seed(42)
        np.random.seed(42)

        output_dir = output_dir_template.format(int(rap_prob * 100))
        print(output_dir)
        for lan_file in lan_eval_files:
            basename = path.basename(lan_file)
            output_file = path.join(output_dir, basename)

            ignore_first = True
            fields = None
            with open(lan_file) as f, open(output_file, 'w') as writer:
                for lan_line in f:
                    if ignore_first:
                        ignore_first = False
                        fields = lan_line.strip().split(",")
                        writer.write(lan_line)
                        # print()
                        continue

                    lan_game_prefix, lan_move_prefix, lan_options_dict = parse_line(lan_line, fields)
                    uci_game_prefix = convert_lan_to_uci(lan_game_prefix)

                    rap_game_prefix = convert_lan_uci_to_rap(lan_game_prefix, uci_game_prefix, rap_prob=rap_prob)

                    instance_str = ' '.join(rap_game_prefix) + " " + lan_move_prefix
                    for field, move_options in lan_options_dict.items():
                        instance_str += f",{move_options}"
                    # print(rap_move_options_dict)
                    writer.write(instance_str + "\n")


def main():
    args = parse_args()

    print("Getting END position eval set")
    get_end_position_eval_data(args)
    print("Getting START position eval set")
    get_start_position_eval_data(args)


if __name__ == '__main__':
    main()
