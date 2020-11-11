import chess
import argparse
import os
from os import path
import numpy as np

from data_processing.get_fen_repr import game_to_parsed_fen_sequence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-base_dir', default=None, type=str,
                        help="Base directory where sub-directories correspond to notation-level directories.")

    args = parser.parse_args()
    return args


def get_fen_for_file(input_file, output_dir):
    basename = path.splitext(path.basename(input_file))[0]
    basename = 'other_eval_easy' if 'easy' in basename else 'other_eval_hard'
    output_file = path.join(output_dir, basename + ".npy")

    all_games = []
    with open(input_file) as reader:
        for idx, line in enumerate(reader):
            if (idx + 1) % 1000 == 0:
                print(f"{idx + 1}th game")
            move_seq = line.strip().split(",")[0].split()[:-1]
            game_array = game_to_parsed_fen_sequence(move_seq)
            all_games.append(game_array)

    np.save(output_file, all_games)


def get_fen_for_dir(input_dir, output_dir):
    input_files = [path.join(input_dir, f'dev_end_actual_{category}.txt') for category in ['easy', 'hard']]
    print(input_files)
    for input_file in input_files:
        print(input_file)
        get_fen_for_file(input_file, output_dir)


def main():
    args = parse_args()
    uci_dir = path.join(args.base_dir, 'other_eval/uci')
    output_dir = path.join(args.base_dir, 'fen')

    if not path.exists(output_dir):
        os.makedirs(output_dir)

    get_fen_for_dir(input_dir=uci_dir, output_dir=output_dir)


if __name__ == '__main__':
    main()

